"""
CLIP Model - Modified v2 (improved regularization and training controls)

Key changes over modified v1:
- Enhanced projection head (3-layer) optional; dropout on by default
- LayerNorm or BatchNorm options (defaults to LayerNorm)
- Progressive unfreezing of ResNet stages (defaults to unfreeze last stage)
- Param grouping helper for differential learning rates (backbone vs head)

Use via train script flag --modified2 (plus --mod_config to pick a preset)
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import CLIPModel

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CLIPImageEncoderV2(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 2048,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
        enhanced_projection: bool = True,
        unfreeze_last_n_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Backbone
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet_output_dim = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()

        # Freeze all, then selectively unfreeze last N stages (layer4, layer3, ...)
        for p in self.resnet.parameters():
            p.requires_grad = False
        stages = [self.resnet.layer4, self.resnet.layer3, self.resnet.layer2, self.resnet.layer1]
        for i in range(min(max(unfreeze_last_n_blocks, 0), 4)):
            for p in stages[i].parameters():
                p.requires_grad = True

        def maybe_norm(dim: int):
            layers: List[nn.Module] = []
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            return layers

        # Projection head
        layers: List[nn.Module] = []
        if enhanced_projection:
            # 2048 -> 2048 -> 1024 -> 512
            layers += [nn.Linear(resnet_output_dim, hidden_dim), *maybe_norm(hidden_dim), nn.GELU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            mid_dim = hidden_dim // 2
            layers += [nn.Linear(hidden_dim, mid_dim), *maybe_norm(mid_dim), nn.GELU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers += [nn.Linear(mid_dim, embedding_dim)]
        else:
            # 2048 -> 2048 -> 512
            layers += [nn.Linear(resnet_output_dim, hidden_dim), *maybe_norm(hidden_dim), nn.GELU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers += [nn.Linear(hidden_dim, embedding_dim)]
        self.projection_head = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.resnet(images)
        emb = self.projection_head(feats)
        return F.normalize(emb, p=2, dim=1)

    def get_param_groups(self, base_lr: float, weight_decay: float, backbone_lr_scale: float = 0.1):
        # Separate backbone vs projection head for differential LRs
        backbone_params = [p for p in self.resnet.parameters() if p.requires_grad]
        head_params = list(self.projection_head.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_scale, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay})
        return groups


class CLIPTextEncoderFrozen(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        for p in self.clip_model.text_model.parameters():
            p.requires_grad = False
        if hasattr(self.clip_model, "text_projection") and self.clip_model.text_projection is not None:
            self.clip_model.text_projection.requires_grad = False
        self.clip_model.text_model.eval()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            txt = out.pooler_output
            if hasattr(self.clip_model, "text_projection") and self.clip_model.text_projection is not None:
                txt = self.clip_model.text_projection(txt)
        return F.normalize(txt, p=2, dim=1)


class CLIPV2(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        use_cached_embeddings: bool = True,
        # mods
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
        enhanced_projection: bool = True,
        unfreeze_last_n_blocks: int = 1,
        modification_name: str = "best_combo_v2",
    ) -> None:
        super().__init__()
        self.use_cached_embeddings = use_cached_embeddings
        self.modification_name = modification_name
        self.image_encoder = CLIPImageEncoderV2(
            embedding_dim=embedding_dim,
            hidden_dim=2048,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            enhanced_projection=enhanced_projection,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        if use_cached_embeddings:
            self.text_encoder = None
        else:
            self.text_encoder = CLIPTextEncoderFrozen(model_name=clip_model_name)

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, input_ids, attention_mask):
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not initialized (using cached embeddings)")
        return self.text_encoder(input_ids, attention_mask)

    def forward(self, images, input_ids, attention_mask):
        return self.encode_image(images), self.encode_text(input_ids, attention_mask)

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_param_groups(self, base_lr: float, weight_decay: float, backbone_lr_scale: float = 0.1):
        return self.image_encoder.get_param_groups(base_lr, weight_decay, backbone_lr_scale)


def get_modification2_configs() -> Dict[str, Dict]:
    return {
        # Good default: LN + Dropout + Enhanced projection + unfreeze layer4
        "best_combo_v2": {
            "use_batch_norm": False,
            "use_layer_norm": True,
            "dropout_rate": 0.15,
            "enhanced_projection": True,
            "unfreeze_last_n_blocks": 1,
        },
        # Stronger regularization, backbone frozen
        "head_only_strong": {
            "use_batch_norm": True,
            "use_layer_norm": False,
            "dropout_rate": 0.2,
            "enhanced_projection": True,
            "unfreeze_last_n_blocks": 0,
        },
        # Train more backbone
        "unfreeze_2_combo": {
            "use_batch_norm": False,
            "use_layer_norm": True,
            "dropout_rate": 0.1,
            "enhanced_projection": True,
            "unfreeze_last_n_blocks": 2,
        },
    }


def create_modified2_clip_model(
    config_name: str = "best_combo_v2",
    embedding_dim: int = None,
    clip_model_name: str = None,
    device: str = "cuda",
    use_cached_embeddings: bool = True,
) -> CLIPV2:
    if embedding_dim is None:
        embedding_dim = config.TEXT_EMBEDDING_DIM
    if clip_model_name is None:
        clip_model_name = config.CLIP_MODEL_NAME
    cfgs = get_modification2_configs()
    if config_name not in cfgs:
        raise ValueError(f"Unknown config: {config_name}. Options: {list(cfgs.keys())}")
    cfg = cfgs[config_name]
    model = CLIPV2(
        embedding_dim=embedding_dim,
        clip_model_name=clip_model_name,
        use_cached_embeddings=use_cached_embeddings,
        modification_name=config_name,
        **cfg,
    )
    return model.to(device)
