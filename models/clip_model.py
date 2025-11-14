"""
CLIP Model Implementation for ELEC475 Lab 4

This module implements a CLIP-style model combining:
- A ResNet50 image encoder (trainable)
- A projection head (trainable)
- A pretrained CLIP text encoder (frozen)

The model is designed to align image and text embeddings in a shared space.
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


class CLIPImageEncoder(nn.Module):
    """
    Image Encoder using ResNet50 with a custom projection head.
    
    Architecture:
    1. ResNet50 backbone (pretrained on ImageNet)
    2. Projection head with two linear layers and GELU activation
    
    TRAINABLE COMPONENTS:
    - ResNet50 backbone: All layers are trainable to allow fine-tuning on COCO
    - Projection head: Trainable to learn optimal mapping to CLIP embedding space
    
    WHY: We train the image encoder because it needs to adapt to the specific
    visual concepts in our dataset and learn to produce embeddings that align
    well with the text representations.
    """
    
    def __init__(self, embedding_dim=512, hidden_dim=2048):
        """
        Initialize the image encoder.
        
        Args:
            embedding_dim (int): Output embedding dimension (matches CLIP text embeddings)
            hidden_dim (int): Hidden dimension for the projection head
        """
        super(CLIPImageEncoder, self).__init__()
        
        # ====================================================================
        # COMPONENT 1: ResNet50 Backbone (TRAINABLE)
        # ====================================================================
        # Load pretrained ResNet50 from torchvision
        # Using pretrained weights provides a strong initialization for visual features
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Get the input dimension of the original final layer
        # ResNet50's final layer (fc) takes 2048-dimensional features
        resnet_output_dim = self.resnet.fc.in_features
        
        # Remove the original classification layer (we'll add our own projection)
        self.resnet.fc = nn.Identity()
        
        # ====================================================================
        # COMPONENT 2: Projection Head (TRAINABLE)
        # ====================================================================
        # Two-layer MLP with GELU activation
        # Maps ResNet features (2048-dim) to CLIP embedding space (512-dim)
        self.projection_head = nn.Sequential(
            nn.Linear(resnet_output_dim, hidden_dim),  # 2048 -> 2048
            nn.GELU(),                                   # Non-linear activation
            nn.Linear(hidden_dim, embedding_dim)        # 2048 -> 512
        )
        
        # WHY THIS ARCHITECTURE:
        # - First linear layer maintains dimensionality for rich feature transformation
        # - GELU provides smooth, non-linear activation (better than ReLU for transformers)
        # - Second linear layer projects to the target embedding dimension
        
    def forward(self, images):
        """
        Forward pass through the image encoder.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: L2-normalized image embeddings of shape (batch_size, 512)
        """
        # Extract visual features using ResNet50 backbone
        # Output shape: (batch_size, 2048)
        features = self.resnet(images)
        
        # Project to CLIP embedding space
        # Output shape: (batch_size, 512)
        embeddings = self.projection_head(features)
        
        # L2 normalize embeddings for cosine similarity computation
        # This ensures embeddings lie on a unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class CLIPTextEncoder(nn.Module):
    """
    Text Encoder using pretrained CLIP model from HuggingFace.
    
    FROZEN COMPONENTS:
    - All parameters of the CLIP text encoder are frozen (requires_grad=False)
    
    WHY FROZEN: The pretrained CLIP text encoder already has excellent text
    understanding capabilities. Freezing it:
    1. Reduces memory usage and training time
    2. Prevents catastrophic forgetting of language understanding
    3. Focuses optimization on the image encoder to match the text space
    4. Is a common practice in CLIP fine-tuning (asymmetric training)
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the text encoder using HuggingFace CLIP.
        
        Args:
            model_name (str): HuggingFace model identifier
        """
        super(CLIPTextEncoder, self).__init__()
        
        # Load pretrained CLIP model from HuggingFace
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        # ====================================================================
        # FREEZE TEXT ENCODER PARAMETERS (NO GRADIENT UPDATES)
        # ====================================================================
        # Set requires_grad=False for all parameters in the text encoder
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        
        # Also freeze the text projection layer
        if hasattr(self.clip_model, 'text_projection'):
            self.clip_model.text_projection.requires_grad = False
        
        # Set model to evaluation mode (disables dropout, batch norm updates)
        self.clip_model.text_model.eval()
        
        print(f"[OK] Loaded pretrained CLIP text encoder: {model_name}")
        print("[OK] All text encoder parameters are FROZEN (no gradient updates)")
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids (torch.Tensor): Tokenized text of shape (batch_size, max_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, max_length)
        
        Returns:
            torch.Tensor: L2-normalized text embeddings of shape (batch_size, 512)
        """
        # Get text features from CLIP model
        # Using torch.no_grad() for efficiency since parameters are frozen
        with torch.no_grad():
            text_outputs = self.clip_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get the pooled output (CLS token representation)
            text_embeddings = text_outputs.pooler_output
            
            # Apply text projection if it exists
            if hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
                text_embeddings = self.clip_model.text_projection(text_embeddings)
        
        # L2 normalize embeddings for cosine similarity computation
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        return text_embeddings


class CLIP(nn.Module):
    """
    Complete CLIP model combining image and text encoders.
    
    TRAINING STRATEGY:
    - Image Encoder: TRAINABLE (adapts to dataset-specific visual features)
    - Projection Head: TRAINABLE (learns optimal mapping to embedding space)
    - Text Encoder: FROZEN (preserves pretrained language understanding)
    
    This asymmetric training approach is efficient and effective for CLIP fine-tuning.
    """
    
    def __init__(self, embedding_dim=512, clip_model_name="openai/clip-vit-base-patch32", 
                 use_cached_embeddings=False):
        """
        Initialize the CLIP model.
        
        Args:
            embedding_dim (int): Dimension of the shared embedding space
            clip_model_name (str): HuggingFace CLIP model name for text encoder
            use_cached_embeddings (bool): If True, skip text encoder initialization to save memory
        """
        super(CLIP, self).__init__()
        
        # Initialize image encoder (TRAINABLE)
        self.image_encoder = CLIPImageEncoder(embedding_dim=embedding_dim)
        
        # Initialize text encoder only if not using cached embeddings
        self.use_cached_embeddings = use_cached_embeddings
        if use_cached_embeddings:
            self.text_encoder = None
            print(f"\n{'='*70}")
            print("CLIP MODEL INITIALIZED (CACHED EMBEDDINGS MODE)")
            print(f"{'='*70}")
            print(f"Embedding dimension: {embedding_dim}")
            print(f"Image encoder: ResNet50 + Projection Head (TRAINABLE)")
            print(f"Text encoder: SKIPPED (using cached embeddings - saves GPU memory)")
            print(f"{'='*70}\n")
        else:
            # Initialize text encoder (FROZEN)
            self.text_encoder = CLIPTextEncoder(model_name=clip_model_name)
            print(f"\n{'='*70}")
            print("CLIP MODEL INITIALIZED")
            print(f"{'='*70}")
            print(f"Embedding dimension: {embedding_dim}")
            print(f"Image encoder: ResNet50 + Projection Head (TRAINABLE)")
            print(f"Text encoder: {clip_model_name} (FROZEN)")
            print(f"{'='*70}\n")
    
    def encode_image(self, images):
        """
        Encode images to embeddings.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: L2-normalized image embeddings of shape (batch_size, embedding_dim)
        """
        return self.image_encoder(images)
    
    def encode_text(self, input_ids, attention_mask):
        """
        Encode text to embeddings.
        
        Args:
            input_ids (torch.Tensor): Tokenized text of shape (batch_size, max_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, max_length)
        
        Returns:
            torch.Tensor: L2-normalized text embeddings of shape (batch_size, embedding_dim)
        
        Raises:
            RuntimeError: If called when using cached embeddings (text encoder not initialized)
        """
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not initialized. This model uses cached embeddings.")
        return self.text_encoder(input_ids, attention_mask)
    
    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass through both encoders.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            input_ids (torch.Tensor): Tokenized text of shape (batch_size, max_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, max_length)
        
        Returns:
            tuple: (image_embeddings, text_embeddings)
                - image_embeddings: L2-normalized, shape (batch_size, embedding_dim)
                - text_embeddings: L2-normalized, shape (batch_size, embedding_dim)
        """
        # Encode images (through trainable encoder)
        image_embeddings = self.encode_image(images)
        
        # Encode text (through frozen encoder)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        return image_embeddings, text_embeddings
    
    def get_trainable_parameters(self):
        """
        Get all trainable parameters for the optimizer.
        
        Returns:
            iterator: Iterator over trainable parameters
        """
        # Only image encoder and projection head parameters are trainable
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def print_parameter_summary(self):
        """Print a summary of trainable vs frozen parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params
        
        print(f"\n{'='*70}")
        print("PARAMETER SUMMARY")
        print(f"{'='*70}")
        print(f"Trainable parameters: {trainable_params:,}")
        if total_params > 0:
            print(f"  ({100*trainable_params/total_params:.2f}% of model parameters)")
        if self.use_cached_embeddings:
            print(f"Frozen parameters:    {frozen_params:,} (text encoder not loaded)")
        else:
            print(f"Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"Total parameters:     {total_params:,}")
        if self.use_cached_embeddings:
            print(f"\n💡 Memory saved: Text encoder (~123M parameters) not loaded")
        print(f"{'='*70}\n")


def create_clip_model(embedding_dim=None, clip_model_name=None, device='cuda', 
                     use_cached_embeddings=False):
    """
    Factory function to create and initialize a CLIP model.
    
    Args:
        embedding_dim (int): Embedding dimension (defaults to config.TEXT_EMBEDDING_DIM)
        clip_model_name (str): CLIP model name (defaults to config.CLIP_MODEL_NAME)
        device (str): Device to place model on ('cuda' or 'cpu')
        use_cached_embeddings (bool): If True, skip text encoder initialization to save memory
    
    Returns:
        CLIP: Initialized CLIP model
    """
    # Use config defaults if not specified
    if embedding_dim is None:
        embedding_dim = config.TEXT_EMBEDDING_DIM
    if clip_model_name is None:
        clip_model_name = config.CLIP_MODEL_NAME
    
    # Create model
    model = CLIP(embedding_dim=embedding_dim, clip_model_name=clip_model_name,
                 use_cached_embeddings=use_cached_embeddings)
    
    # Move to device
    model = model.to(device)
    
    # Print parameter summary
    model.print_parameter_summary()
    
    return model


if __name__ == "__main__":
    """
    Test script to verify model initialization and forward pass.
    """
    print("Testing CLIP model implementation...\n")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model = create_clip_model(device=device)
    
    # Create dummy inputs
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_input_ids = torch.randint(0, 49408, (batch_size, 77)).to(device)
    dummy_attention_mask = torch.ones(batch_size, 77).to(device)
    
    print("Testing forward pass...")
    print(f"Input shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Input IDs: {dummy_input_ids.shape}")
    print(f"  Attention Mask: {dummy_attention_mask.shape}\n")
    
    # Forward pass
    with torch.no_grad():
        image_embeds, text_embeds = model(dummy_images, dummy_input_ids, dummy_attention_mask)
    
    print(f"Output shapes:")
    print(f"  Image embeddings: {image_embeds.shape}")
    print(f"  Text embeddings: {text_embeds.shape}\n")
    
    # Verify normalization
    image_norms = torch.norm(image_embeds, dim=1)
    text_norms = torch.norm(text_embeds, dim=1)
    
    print("Verifying L2 normalization:")
    print(f"  Image embedding norms: {image_norms.cpu().numpy()}")
    print(f"  Text embedding norms: {text_norms.cpu().numpy()}")
    print(f"  All norms ~= 1.0: {torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-5)}")
    
    print("\n[OK] Model implementation test complete!")

