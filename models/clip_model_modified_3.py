"""
CLIP Model - Modified v3 (backbone options, residual projection, stronger regularization)

Goals:
- Improve retrieval (Recall@K) and reduce overfitting vs v1/v2
- Allow swapping stronger CNN backbones without new deps (torchvision only)
- Use residual LayerNorm projection head with dropout
- Support progressive unfreezing and differential learning rates
- Keep cached-text-embeddings workflow intact

Usage:
- In train script, import create_modified3_clip_model and pass --modified3 (add flag) or swap call
- Defaults target balanced stability/performance on MPS
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
)

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# --------- Helpers ---------
class GeM(nn.Module):
    """Generalized Mean Pooling.
    If input is already pooled to a vector, GeM is a no-op. We apply GeM only when 4D inputs are given.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.clamp(min=self.eps).pow(self.p)
            x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
            x = x.pow(1.0 / self.p)
            return x.view(x.size(0), -1)
        # If already flattened [B, C], return as-is
        return x


def remove_classifier(module: nn.Module) -> Tuple[nn.Module, int, bool]:
    """Remove the classifier head and return (feature_extractor, feature_dim, expects_4d).
    expects_4d indicates whether the backbone forward outputs a spatial map (for GeM).
    """
    # ResNet50
    if hasattr(module, 'fc') and hasattr(module, 'avgpool'):
        feat_dim = module.fc.in_features
        module.fc = nn.Identity()
        # ResNet forward returns pooled vector; we keep it as vector (expects_4d=False)
        return module, feat_dim, False
    # ConvNeXt Tiny
    if hasattr(module, 'classifier') and hasattr(module, 'features'):
        # classifier often is Sequential(LayerNorm2d, Flatten, Linear(in_features, num_classes))
        feat_dim = None
        if isinstance(module.classifier, nn.Sequential):
            for m in module.classifier:
                if isinstance(m, nn.Linear):
                    feat_dim = m.in_features
                    break
        # Fallback: derive from last feature stage channels
        if feat_dim is None:
            try:
                last_block = list(module.features.children())[-1]
                # ConvNeXt block has out_channels attribute on last layer
                feat_dim = getattr(last_block, 'out_channels', None)
            except Exception:
                feat_dim = None
        if feat_dim is None:
            raise ValueError('Could not determine ConvNeXt feature dim')
        module.classifier = nn.Identity()
        # ConvNeXt forward returns avgpooled 4D tensor [B,C,1,1]
        return module, feat_dim, True
    # EfficientNet-B3
    if hasattr(module, 'classifier'):
        try:
            feat_dim = module.classifier[1].in_features
        except Exception:
            feat_dim = module.classifier.in_features
        module.classifier = nn.Identity()
        # EfficientNet forward returns avgpooled 4D tensor [B,C,1,1]
        return module, feat_dim, True
    raise ValueError('Unsupported backbone structure')


class ResidualProjectionHead(nn.Module):
    """Projection head with residual skip and LayerNorm.
    Maps feature_dim -> 1024 -> 512 with dropout, plus 1x1 skip from feature_dim -> 512.
    """
    def __init__(self, feature_dim: int, embed_dim: int = 512, mid_dim: int = 1024, dropout: float = 0.2,
                 use_layer_norm: bool = True, use_batch_norm: bool = False):
        super().__init__()
        self.use_ln = use_layer_norm
        self.use_bn = use_batch_norm
        layers: List[nn.Module] = [
            nn.Linear(feature_dim, mid_dim),
        ]
        if self.use_bn:
            layers.append(nn.BatchNorm1d(mid_dim))
        if self.use_ln:
            layers.append(nn.LayerNorm(mid_dim))
        layers += [
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, embed_dim),
        ]
        self.main = nn.Sequential(*layers)
        self.skip = nn.Linear(feature_dim, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim) if self.use_ln else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main(x)
        s = self.skip(x)
        out = y + s
        out = self.out_norm(out)
        return out


# --------- Image Encoder v3 ---------
class CLIPImageEncoderV3(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        backbone_name: str = 'resnet50',
        dropout_rate: float = 0.2,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        unfreeze_last_n_blocks: int = 1,
        use_gem: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_gem = use_gem

        # Build backbone
        if backbone_name == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone, feat_dim, expects_4d = remove_classifier(backbone)
            # Unfreeze stages: layer4, layer3, layer2, layer1
            for p in self.backbone.parameters():
                p.requires_grad = False
            stages = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
            for i in range(min(max(unfreeze_last_n_blocks, 0), 4)):
                for p in stages[i].parameters():
                    p.requires_grad = True
        elif backbone_name == 'convnext_tiny':
            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.backbone, feat_dim, expects_4d = remove_classifier(backbone)
            # ConvNeXt: features are grouped; simply unfreeze last N blocks in features
            for p in self.backbone.parameters():
                p.requires_grad = False
            blocks = list(self.backbone.features.children())[::-1]
            unfrozen = 0
            for block in blocks:
                if unfrozen >= unfreeze_last_n_blocks:
                    break
                for p in block.parameters():
                    p.requires_grad = True
                unfrozen += 1
        elif backbone_name == 'efficientnet_b3':
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.backbone, feat_dim, expects_4d = remove_classifier(backbone)
            for p in self.backbone.parameters():
                p.requires_grad = False
            # unfreeze last N blocks (approx: last N children in features)
            try:
                blocks = list(self.backbone.features.children())[::-1]
                unfrozen = 0
                for block in blocks:
                    if unfrozen >= unfreeze_last_n_blocks:
                        break
                    for p in block.parameters():
                        p.requires_grad = True
                    unfrozen += 1
            except Exception:
                pass
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.expects_4d = expects_4d
        self.pool = GeM() if use_gem else nn.Identity()

        # Projection head: residual + LayerNorm + dropout
        self.projection_head = ResidualProjectionHead(
            feature_dim=feat_dim,
            embed_dim=embedding_dim,
            mid_dim=1024,
            dropout=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        # If backbone returned spatial map, pool to vector; else assume vector
        if feats.dim() == 4:
            feats = self.pool(feats)
            feats = feats.view(feats.size(0), -1)
        emb = self.projection_head(feats)
        return F.normalize(emb, p=2, dim=1)

    def get_param_groups(self, base_lr: float, weight_decay: float, backbone_lr_scale: float = 0.2):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.projection_head.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_scale, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay})
        return groups


# --------- Full CLIP v3 (image-only training with cached text) ---------
class CLIPV3(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        use_cached_embeddings: bool = True,
        backbone_name: str = 'resnet50',
        dropout_rate: float = 0.2,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        unfreeze_last_n_blocks: int = 1,
        use_gem: bool = False,
        modification_name: str = 'best_v3_default',
    ) -> None:
        super().__init__()
        self.use_cached_embeddings = use_cached_embeddings
        self.modification_name = modification_name
        self.image_encoder = CLIPImageEncoderV3(
            embedding_dim=embedding_dim,
            backbone_name=backbone_name,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            use_gem=use_gem,
        )
        self.text_encoder = None  # cached embeddings mode only

    def encode_image(self, images):
        return self.image_encoder(images)

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_param_groups(self, base_lr: float, weight_decay: float, backbone_lr_scale: float = 0.2):
        return self.image_encoder.get_param_groups(base_lr, weight_decay, backbone_lr_scale)


def get_modification3_configs() -> Dict[str, Dict]:
    return {
        # Balanced default for MPS: ResNet50, residual LN head, dropout 0.2, unfreeze last stage
        'best_v3_default': {
            'backbone_name': 'resnet50',
            'dropout_rate': 0.2,
            'use_layer_norm': True,
            'use_batch_norm': False,
            'unfreeze_last_n_blocks': 1,
            'use_gem': False,
        },
        # Stronger backbone, slightly lower dropout
        'convnext_tiny_v3': {
            'backbone_name': 'convnext_tiny',
            'dropout_rate': 0.15,
            'use_layer_norm': True,
            'use_batch_norm': False,
            'unfreeze_last_n_blocks': 2,
            'use_gem': False,
        },
        # EfficientNet-B3 with GeM pooling (often helps retrieval)
        'efficientnet_b3_v3': {
            'backbone_name': 'efficientnet_b3',
            'dropout_rate': 0.15,
            'use_layer_norm': True,
            'use_batch_norm': False,
            'unfreeze_last_n_blocks': 2,
            'use_gem': True,
        },
    }


def create_modified3_clip_model(
    config_name: str = 'best_v3_default',
    embedding_dim: int = None,
    device: str = 'cuda',
    use_cached_embeddings: bool = True,
) -> CLIPV3:
    if embedding_dim is None:
        embedding_dim = config.TEXT_EMBEDDING_DIM
    cfgs = get_modification3_configs()
    if config_name not in cfgs:
        raise ValueError(f"Unknown config: {config_name}. Options: {list(cfgs.keys())}")
    cfg = cfgs[config_name]
    model = CLIPV3(
        embedding_dim=embedding_dim,
        use_cached_embeddings=use_cached_embeddings,
        modification_name=config_name,
        **cfg,
    )
    return model.to(device)
