# Models Directory

This directory contains the model implementations for ELEC475 Lab 4.

## CLIP Model (`clip_model.py`)

### Overview
Implementation of a CLIP-style model for image-text alignment using:
- **Image Encoder**: ResNet50 (pretrained) + 2-layer projection head
- **Text Encoder**: Pretrained CLIP ViT-B/32 from HuggingFace (frozen)

### Key Classes

#### `CLIPImageEncoder`
- Uses `torchvision.models.resnet50` with ImageNet pretrained weights
- Replaces final FC layer with a 2-layer projection head (with GELU)
- Output: 512-dimensional L2-normalized embeddings
- **Status**: Trainable

#### `CLIPTextEncoder`
- Loads `openai/clip-vit-base-patch32` from HuggingFace
- All parameters frozen (`requires_grad=False`)
- Output: 512-dimensional L2-normalized embeddings
- **Status**: Frozen (no gradient updates)

#### `CLIP`
- Main model combining both encoders
- Methods:
  - `encode_image(images)`: Encode images to embeddings
  - `encode_text(input_ids, attention_mask)`: Encode text to embeddings
  - `forward(images, input_ids, attention_mask)`: Encode both modalities
  - `get_trainable_parameters()`: Get parameters for optimizer
  - `print_parameter_summary()`: Display trainable vs frozen parameter counts

### Usage Example

```python
from models.clip_model import create_clip_model
import torch

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_clip_model(device=device)

# Prepare inputs
images = torch.randn(32, 3, 224, 224).to(device)
input_ids = torch.randint(0, 49408, (32, 77)).to(device)
attention_mask = torch.ones(32, 77).to(device)

# Forward pass
image_embeds, text_embeds = model(images, input_ids, attention_mask)

# Both embeddings are L2-normalized and have shape (32, 512)
print(image_embeds.shape)  # torch.Size([32, 512])
print(text_embeds.shape)   # torch.Size([32, 512])
```

### Architecture Details

**Image Path:**
```
Input (224×224×3) 
  → ResNet50 (trainable) 
  → Features (2048-dim)
  → Linear (2048→2048, trainable) 
  → GELU
  → Linear (2048→512, trainable)
  → L2 Normalize
  → Output (512-dim)
```

**Text Path:**
```
Input (tokenized text)
  → CLIP ViT-B/32 (frozen)
  → Text Features (512-dim)
  → L2 Normalize
  → Output (512-dim)
```

### Training Philosophy

- **Asymmetric Training**: Only the image encoder is trained
- **Frozen Text Encoder**: Preserves pretrained language understanding
- **Benefits**: 
  - Faster training (only ~30% of parameters updated)
  - Lower memory usage
  - Better stability and convergence
  - Maintains CLIP's strong text representations

See `analysis/readme.md` for detailed explanation of trainable vs frozen components.