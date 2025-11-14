# ELEC 475 Lab 4 - Analysis & Documentation

## Step 1: Dataset Preparation

### Dataset Split Sizes
- **Subset of COCO 2014 dataset**
- Train: 16,000 images
- Validation: 4,000 images
- Total: 20,000 images (with 5 captions each)

### Image Preprocessing
- **Resize**: 224 × 224 pixels
- **Normalization**: 
  - Mean: `[0.48145466, 0.4578275, 0.40821073]`
  - Std: `[0.26862954, 0.26130258, 0.27577711]`

### Text Preprocessing
- **Tokenization**: CLIP BPE tokenizer (`openai/clip-vit-base-patch32`)
- **Max Length**: 77 tokens
- **Padding Strategy**: `padding='max_length'` (pad all sequences to 77 tokens)
- **Truncation**: `truncation=True` (truncate sequences exceeding max length)

---

## Step 2: Model Design (CLIP Architecture)

### Model Architecture Overview
Our CLIP model consists of two encoders that map images and text into a shared 512-dimensional embedding space:

1. **Image Encoder**: ResNet50 + Projection Head
2. **Text Encoder**: Pretrained CLIP ViT-B/32 from HuggingFace

### Image Encoder Architecture

**Components:**
- **ResNet50 Backbone** (pretrained on ImageNet)
  - Input: 224×224×3 images
  - Output: 2048-dimensional feature vectors
  - Pretrained weights: `ResNet50_Weights.IMAGENET1K_V2`

- **Projection Head** (2-layer MLP)
  - Layer 1: Linear(2048 → 2048) + GELU
  - Layer 2: Linear(2048 → 512)
  - Output: L2-normalized 512-dimensional embeddings

### Text Encoder Architecture

**Components:**
- **Pretrained CLIP ViT-B/32** from HuggingFace
  - Model: `openai/clip-vit-base-patch32`
  - Input: Tokenized text (max 77 tokens)
  - Output: L2-normalized 512-dimensional embeddings

---

## Trainable vs Frozen Components

### ✅ TRAINABLE Components

#### 1. ResNet50 Image Encoder (All Layers)
**Why trainable?**
- Needs to adapt to the specific visual concepts in COCO dataset
- Fine-tuning allows the model to learn domain-specific visual features
- ImageNet pretraining provides good initialization, but task-specific adaptation improves performance
- Enables the model to learn which visual features are most relevant for text-image alignment

#### 2. Projection Head (Both Linear Layers)
**Why trainable?**
- Learns optimal mapping from ResNet features to CLIP embedding space
- Must adapt to align with the frozen text encoder's representation
- Critical for bridging the gap between ImageNet features and CLIP's multimodal space
- Only ~6M parameters, so computationally cheap to train

**Total Trainable Parameters:** ~28M (ResNet50: ~23M + Projection: ~6M)

---

### ❄️ FROZEN Components

#### Text Encoder (All Parameters)
**Why frozen?**

1. **Preserves Language Understanding**
   - The pretrained CLIP text encoder has excellent semantic understanding of natural language
   - Freezing prevents "catastrophic forgetting" of language knowledge
   - Text encoder was trained on 400M image-text pairs, far more than our dataset

2. **Computational Efficiency**
   - Reduces memory usage (no gradients stored for text encoder)
   - Speeds up training (fewer parameters to update)
   - Allows larger batch sizes

3. **Asymmetric Training Strategy**
   - Common practice in CLIP fine-tuning
   - We adapt the image encoder to match the text space, not vice versa
   - The text space is already well-structured and semantically meaningful

4. **Stability**
   - Fixing one encoder provides a stable target for the other encoder
   - Reduces risk of both encoders drifting in unproductive directions
   - Helps training converge faster and more reliably

**Total Frozen Parameters:** ~63M (CLIP ViT-B/32 text encoder)

---

### Training Philosophy

**Asymmetric Fine-tuning Approach:**
- We train the **vision side** to align with the **frozen language side**
- This is more efficient than joint training of both encoders
- The pretrained text encoder provides a strong, stable embedding space
- The image encoder learns to map visual concepts into this existing space

**Benefits:**
1. Faster convergence (only ~30% of parameters being trained)
2. Lower memory requirements
3. Better generalization (leverages strong pretrained text representations)
4. Maintains zero-shot capabilities of original CLIP text encoder

---

### Embedding Normalization

Both image and text embeddings are **L2-normalized** before use:
- Ensures embeddings lie on a unit hypersphere
- Cosine similarity becomes equivalent to dot product
- Stabilizes contrastive learning
- Standard practice in metric learning and CLIP training

**Formula:** `normalized_embedding = embedding / ||embedding||₂`

---

### Model Summary

| Component | Parameters | Status | Purpose |
|-----------|-----------|--------|---------|
| ResNet50 Backbone | ~23M | ✅ Trainable | Extract visual features |
| Projection Head | ~6M | ✅ Trainable | Map to CLIP space |
| CLIP Text Encoder | ~63M | ❄️ Frozen | Encode text semantics |
| **Total** | **~92M** | **~30% trainable** | **Multimodal alignment** |