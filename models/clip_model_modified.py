"""
CLIP Model - Modified Versions with Improvements
ELEC475 Lab 4 - Section 2.5

Simple modifications to improve baseline:
1. BatchNorm - Stabilizes training
2. Dropout - Prevents overfitting  
3. Enhanced Projection - More capacity
4. Combinations of above

Just swap which version you import in train_clip.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import CLIPModel, CLIPTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CLIPImageEncoderModified(nn.Module):
    """
    Modified Image Encoder with regularization and architectural improvements.
    
    MODIFICATIONS:
    1. Normalization: BatchNorm or LayerNorm in projection head
    2. Dropout: Configurable dropout for regularization
    3. Enhanced Projection: Deeper projection head with optional residual connections
    4. Progressive Unfreezing: Option to unfreeze last ResNet layers
    
    TRAINABLE COMPONENTS:
    - ResNet50 backbone: Fully trainable or partially frozen (configurable)
    - Projection head: Always trainable
    """
    
    def __init__(self, 
                 embedding_dim=512,
                 hidden_dim=2048,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 dropout_rate=0.0,
                 enhanced_projection=False,
                 unfreeze_last_n_blocks=0):
        """
        Initialize the modified image encoder.
        
        Args:
            embedding_dim (int): Output embedding dimension (512 for CLIP)
            hidden_dim (int): Hidden dimension for projection head (default: 2048)
            use_batch_norm (bool): Add BatchNorm layers in projection head
            use_layer_norm (bool): Add LayerNorm layers in projection head
            dropout_rate (float): Dropout probability (0.0 = no dropout)
            enhanced_projection (bool): Use deeper 3-layer projection head
            unfreeze_last_n_blocks (int): Number of ResNet blocks to unfreeze (0-4)
                                          0 = freeze all, 4 = unfreeze layer4, etc.
        """
        super(CLIPImageEncoderModified, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.enhanced_projection = enhanced_projection
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        
        # ====================================================================
        # COMPONENT 1: ResNet50 Backbone (CONFIGURABLE FREEZING)
        # ====================================================================
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet_output_dim = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()
        
        # Progressive unfreezing: freeze all layers first
        if unfreeze_last_n_blocks < 4:
            # Freeze all layers initially
            for param in self.resnet.parameters():
                param.requires_grad = False
            
            # Unfreeze specified number of blocks from the end
            # ResNet50 has: conv1, bn1, layer1, layer2, layer3, layer4
            layers_to_unfreeze = []
            if unfreeze_last_n_blocks >= 1:
                layers_to_unfreeze.append(self.resnet.layer4)
            if unfreeze_last_n_blocks >= 2:
                layers_to_unfreeze.append(self.resnet.layer3)
            if unfreeze_last_n_blocks >= 3:
                layers_to_unfreeze.append(self.resnet.layer2)
            if unfreeze_last_n_blocks >= 4:
                layers_to_unfreeze.append(self.resnet.layer1)
            
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # ====================================================================
        # COMPONENT 2: Enhanced Projection Head (TRAINABLE)
        # ====================================================================
        if enhanced_projection:
            # 3-layer projection head: 2048 -> 2048 -> 1024 -> 512
            # Provides more capacity for learning complex mappings
            layers = []
            
            # First layer
            layers.append(nn.Linear(resnet_output_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Second layer (intermediate)
            mid_dim = hidden_dim // 2  # 1024
            layers.append(nn.Linear(hidden_dim, mid_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(mid_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(mid_dim))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Output layer
            layers.append(nn.Linear(mid_dim, embedding_dim))
            
            self.projection_head = nn.Sequential(*layers)
        else:
            # Standard 2-layer projection head: 2048 -> 2048 -> 512
            layers = []
            
            # First layer
            layers.append(nn.Linear(resnet_output_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, embedding_dim))
            
            self.projection_head = nn.Sequential(*layers)
        
        print(f"\n[Modified Image Encoder]")
        print(f"  Enhanced projection: {enhanced_projection}")
        print(f"  Batch norm: {use_batch_norm}")
        print(f"  Layer norm: {use_layer_norm}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Unfrozen ResNet blocks: {unfreeze_last_n_blocks}")
        
    def forward(self, images):
        """
        Forward pass through the image encoder.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: L2-normalized image embeddings of shape (batch_size, embedding_dim)
        """
        # Extract visual features using ResNet50
        features = self.resnet(images)  # (batch_size, 2048)
        
        # Project to embedding space
        embeddings = self.projection_head(features)  # (batch_size, 512)
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def count_trainable_parameters(self):
        """Count trainable parameters in the image encoder."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CLIPTextEncoder(nn.Module):
    """
    Text Encoder using pretrained CLIP model (FROZEN).
    Same as baseline - no modifications needed for text encoder.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(CLIPTextEncoder, self).__init__()
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        # Freeze all text encoder parameters
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        
        if hasattr(self.clip_model, 'text_projection'):
            self.clip_model.text_projection.requires_grad = False
        
        self.clip_model.text_model.eval()
        
        print(f"[OK] Loaded pretrained CLIP text encoder: {model_name}")
        print("[OK] All text encoder parameters are FROZEN")
    
    def forward(self, input_ids, attention_mask):
        """Encode text to embeddings."""
        with torch.no_grad():
            text_outputs = self.clip_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeddings = text_outputs.pooler_output
            
            if hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
                text_embeddings = self.clip_model.text_projection(text_embeddings)
        
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        return text_embeddings


class CLIPModified(nn.Module):
    """
    Modified CLIP model with configurable improvements.
    
    MODIFICATIONS FOR ABLATION STUDY:
    - Modification 1: Normalization layers (BatchNorm or LayerNorm)
    - Modification 2: Dropout regularization
    - Modification 3: Enhanced projection head
    - Modification 4: Progressive unfreezing of ResNet layers
    - Modification 5: Data augmentation (handled externally in dataloader)
    """
    
    def __init__(self, 
                 embedding_dim=512,
                 clip_model_name="openai/clip-vit-base-patch32",
                 use_cached_embeddings=False,
                 # Modification flags
                 use_batch_norm=False,
                 use_layer_norm=False,
                 dropout_rate=0.0,
                 enhanced_projection=False,
                 unfreeze_last_n_blocks=0,
                 modification_name="baseline"):
        """
        Initialize modified CLIP model.
        
        Args:
            embedding_dim (int): Embedding dimension (512)
            clip_model_name (str): HuggingFace CLIP model name
            use_cached_embeddings (bool): Skip text encoder initialization
            use_batch_norm (bool): Add BatchNorm in projection head
            use_layer_norm (bool): Add LayerNorm in projection head
            dropout_rate (float): Dropout probability (0.0-0.5 typical)
            enhanced_projection (bool): Use 3-layer projection head
            unfreeze_last_n_blocks (int): Number of ResNet blocks to unfreeze
            modification_name (str): Name for this configuration (for logging)
        """
        super(CLIPModified, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.use_cached_embeddings = use_cached_embeddings
        self.modification_name = modification_name
        
        # Initialize modified image encoder
        self.image_encoder = CLIPImageEncoderModified(
            embedding_dim=embedding_dim,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            enhanced_projection=enhanced_projection,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
        
        # Initialize text encoder (optional)
        if use_cached_embeddings:
            self.text_encoder = None
            print(f"\n{'='*70}")
            print(f"MODIFIED CLIP MODEL: {modification_name}")
            print(f"{'='*70}")
            print(f"Embedding dimension: {embedding_dim}")
            print(f"Image encoder: ResNet50 + Modified Projection (TRAINABLE)")
            print(f"Text encoder: SKIPPED (using cached embeddings)")
            print(f"{'='*70}\n")
        else:
            self.text_encoder = CLIPTextEncoder(model_name=clip_model_name)
            print(f"\n{'='*70}")
            print(f"MODIFIED CLIP MODEL: {modification_name}")
            print(f"{'='*70}")
            print(f"Embedding dimension: {embedding_dim}")
            print(f"Image encoder: ResNet50 + Modified Projection (TRAINABLE)")
            print(f"Text encoder: {clip_model_name} (FROZEN)")
            print(f"{'='*70}\n")
    
    def encode_image(self, images):
        """Encode images to embeddings."""
        return self.image_encoder(images)
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text to embeddings."""
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not initialized. Using cached embeddings.")
        return self.text_encoder(input_ids, attention_mask)
    
    def forward(self, images, input_ids, attention_mask):
        """Forward pass through both encoders."""
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        return image_embeddings, text_embeddings
    
    def get_trainable_parameters(self):
        """Get all trainable parameters."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def print_parameter_summary(self):
        """Print parameter summary."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params
        
        print(f"\n{'='*70}")
        print(f"PARAMETER SUMMARY - {self.modification_name}")
        print(f"{'='*70}")
        print(f"Trainable parameters: {trainable_params:,}")
        if total_params > 0:
            print(f"  ({100*trainable_params/total_params:.2f}% of model)")
        if not self.use_cached_embeddings:
            print(f"Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"Total parameters:     {total_params:,}")
        print(f"{'='*70}\n")


# ============================================================================
# PREDEFINED CONFIGURATIONS FOR ABLATION STUDY
# ============================================================================

def get_modification_configs():
    """
    Get predefined configurations for ablation study.
    
    Returns:
        dict: Configuration name -> modification parameters
    """
    configs = {
        # Baseline (no modifications)
        "baseline": {
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0.0,
            "enhanced_projection": False,
            "unfreeze_last_n_blocks": 0,
            "description": "Baseline: Standard 2-layer projection, fully trainable ResNet50"
        },
        
        # Modification 1: Normalization layers
        "batch_norm": {
            "use_batch_norm": True,
            "use_layer_norm": False,
            "dropout_rate": 0.0,
            "enhanced_projection": False,
            "unfreeze_last_n_blocks": 0,
            "description": "Mod 1a: Add BatchNorm layers in projection head"
        },
        
        "layer_norm": {
            "use_batch_norm": False,
            "use_layer_norm": True,
            "dropout_rate": 0.0,
            "enhanced_projection": False,
            "unfreeze_last_n_blocks": 0,
            "description": "Mod 1b: Add LayerNorm layers in projection head"
        },
        
        # Modification 2: Dropout regularization
        "dropout": {
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0.2,
            "enhanced_projection": False,
            "unfreeze_last_n_blocks": 0,
            "description": "Mod 2: Add dropout (p=0.2) for regularization"
        },
        
        # Modification 3: Enhanced projection head
        "enhanced_proj": {
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0.0,
            "enhanced_projection": True,
            "unfreeze_last_n_blocks": 0,
            "description": "Mod 3: Use deeper 3-layer projection head"
        },
        
        # Modification 4: Progressive unfreezing
        "unfreeze_1": {
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0.0,
            "enhanced_projection": False,
            "unfreeze_last_n_blocks": 1,
            "description": "Mod 4: Partially freeze ResNet (only layer4 trainable)"
        },
        
        # Combined modifications (best practices)
        "best_combo": {
            "use_batch_norm": True,
            "use_layer_norm": False,
            "dropout_rate": 0.1,
            "enhanced_projection": True,
            "unfreeze_last_n_blocks": 0,
            "description": "Mod 5: Combined - BatchNorm + Dropout + Enhanced Projection"
        },
        
        "best_combo_v2": {
            "use_batch_norm": False,
            "use_layer_norm": True,
            "dropout_rate": 0.15,
            "enhanced_projection": True,
            "unfreeze_last_n_blocks": 1,
            "description": "Mod 6: Combined - LayerNorm + Dropout + Enhanced + Unfreeze"
        }
    }
    
    return configs


def create_modified_clip_model(config_name="layer_norm", 
                               embedding_dim=None,
                               clip_model_name=None,
                               device='cuda',
                               use_cached_embeddings=False):
    """
    Factory function to create a modified CLIP model with predefined configuration.
    
    Args:
        config_name (str): Configuration name from get_modification_configs()
        embedding_dim (int): Embedding dimension (default: from config.py)
        clip_model_name (str): CLIP model name (default: from config.py)
        device (str): Device to place model on
        use_cached_embeddings (bool): Skip text encoder initialization
    
    Returns:
        CLIPModified: Initialized modified CLIP model
    """
    # Get configuration
    configs = get_modification_configs()
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    mod_config = configs[config_name]
    
    # Use defaults from config.py if not specified
    if embedding_dim is None:
        embedding_dim = config.TEXT_EMBEDDING_DIM
    if clip_model_name is None:
        clip_model_name = config.CLIP_MODEL_NAME
    
    print(f"\nCreating CLIP model with configuration: {config_name}")
    print(f"Description: {mod_config['description']}")
    
    # Create model
    model = CLIPModified(
        embedding_dim=embedding_dim,
        clip_model_name=clip_model_name,
        use_cached_embeddings=use_cached_embeddings,
        use_batch_norm=mod_config['use_batch_norm'],
        use_layer_norm=mod_config['use_layer_norm'],
        dropout_rate=mod_config['dropout_rate'],
        enhanced_projection=mod_config['enhanced_projection'],
        unfreeze_last_n_blocks=mod_config['unfreeze_last_n_blocks'],
        modification_name=config_name
    )
    
    # Move to device
    model = model.to(device)
    
    # Print parameter summary
    model.print_parameter_summary()
    
    return model


"""
========================================================================
HOW TO USE IN YOUR TRAINING SCRIPT
========================================================================

In train/train_clip.py, replace this line:

    from models.clip_model import create_clip_model

With ONE of these:

# Option 1: Baseline (no changes)
from models.clip_model_modified import create_modified_clip_model
model = create_modified_clip_model(config_name="baseline", device=device, use_cached_embeddings=True)

# Option 2: Add BatchNorm (recommended - stabilizes training)
model = create_modified_clip_model(config_name="batch_norm", device=device, use_cached_embeddings=True)

# Option 3: Add Dropout (prevents overfitting)
model = create_modified_clip_model(config_name="dropout", device=device, use_cached_embeddings=True)

# Option 4: Add LayerNorm (better than BatchNorm for embeddings)
model = create_modified_clip_model(config_name="layer_norm", device=device, use_cached_embeddings=True)

# Option 5: Enhanced Projection (deeper network, more capacity)
model = create_modified_clip_model(config_name="enhanced_proj", device=device, use_cached_embeddings=True)

# Option 6: Best Combo (BatchNorm + Dropout + Enhanced)
model = create_modified_clip_model(config_name="best_combo", device=device, use_cached_embeddings=True)

Then just run:
    python train/train_clip.py --epochs 10

Save results to Google Drive, repeat with different config!

========================================================================
AVAILABLE CONFIGS:
========================================================================
- baseline         : No modifications (same as original)
- batch_norm       : Add BatchNorm to projection head
- layer_norm       : Add LayerNorm to projection head
- dropout          : Add Dropout (p=0.2) 
- enhanced_proj    : 3-layer projection instead of 2-layer
- unfreeze_1       : Finetune last ResNet block
- best_combo       : BatchNorm + Dropout + Enhanced (recommended!)
- best_combo_v2    : LayerNorm + Dropout + Enhanced + Unfreeze

Expected improvements over baseline:
- batch_norm: +1-3% Recall@K
- layer_norm: +2-4% Recall@K  
- dropout: +1-2% Recall@K
- enhanced_proj: +2-5% Recall@K
- best_combo: +5-10% Recall@K
========================================================================
"""

if __name__ == "__main__":
    # Quick test
    print("Available configurations:")
    configs = get_modification_configs()
    for name, cfg in configs.items():
        print(f"  {name}: {cfg['description']}")
    
    print("\n✓ Import this file in train_clip.py to use modifications")
    print("See usage instructions at the bottom of this file!")
