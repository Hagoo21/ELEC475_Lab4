# ELEC475_Lab4


# ELEC 475 Lab 4 - Section 2.1 Dataset Preparation (MS COCO)
# Goal: Prepare the COCO 2014 dataset for CLIP fine-tuning.

Generate a complete Python script using PyTorch and FiftyOne that:

### 1. Downloads and loads a subset of the COCO 2014 dataset using FiftyOne:
   - Use: `foz.load_zoo_dataset("coco-2014", split="train", label_types=["captions"])`
   - After loading, randomly sample **exactly 2000 images** from the training split.
   - Do the same for the validation split: load val split and sample exactly 2000 images.
   - Export each sampled split to a local folder with this structure:
        data/coco_subset/train/images/
        data/coco_subset/train/captions.json
        data/coco_subset/val/images/
        data/coco_subset/val/captions.json
   - Ensure `captions.json` follows the COCO caption format (image_id, caption list, etc.)

### 2. Implement CLIP-style image preprocessing:
   - Resize all images to 224x224
   - Normalize using CLIP's statistics:
       mean=[0.48145466, 0.4578275, 0.40821073]
       std=[0.26862954, 0.26130258, 0.27577711]

### 3. Use the pretrained CLIP text encoder:
   - Load "openai/clip-vit-base-patch32" via HuggingFace
   - Tokenize captions (document max_length, truncation, and padding strategy)
   - Encode captions and save embeddings

### 4. Implement a PyTorch dataset class that returns:
   - normalized image tensor
   - precomputed text embedding (loaded from cache)
   - image_id

### 5. Add caching functionality:
   - Save training caption embeddings to: `train_text_embeds.pt`
   - Save validation caption embeddings to: `val_text_embeds.pt`
   - Ensure that on subsequent runs, embeddings are loaded from cache instead of recomputed.

### 6. Verification:
   - Randomly pick 5 samples
   - Display both the image and its raw caption (pre-tokenization)
   - Print tensor shapes for image tensors and text embedding tensors

### 7. Documentation (as code comments):
   - Exact dataset split sizes (2000 train, 2000 val)
   - Full image preprocessing pipeline (resize → normalize)
   - Text preprocessing method:
       - tokenizer used
       - max_length
       - truncation strategy
       - padding method
   - Explanation of why caching text embeddings improves speed and reduces GPU memory usage.

### Requirements:
   - Use torchvision, PyTorch, HuggingFace Transformers, and FiftyOne
   - Include a main() function that prepares the dataset end-to-end
   - Ensure deterministic sampling with a fixed random seed

Generate the full script with imports, helper utilities, and a runnable main() block.



# ELEC 475 Lab 4 - Section 2.2 Model Design (CLIP)

Generate PyTorch code for the CLIP model structure:
1. Implement an image encoder using torchvision.models.resnet50(pretrained=True).
2. Replace the final layer with a projection head:
   - Two Linear layers with GELU activation in between
   - Output dimension = 512 (to match CLIP text embedding size)
3. Load the pretrained CLIP text encoder from HuggingFace ("openai/clip-vit-base-patch32").
4. Freeze all parameters of the text encoder (no gradient updates).
5. Ensure the image encoder and projection layers are trainable.
6. Return normalized image and text embeddings (L2-normalized).
7. Write clear comments explaining which parts are trainable and why.



# ELEC 475 Lab 4 - Section 2.3 Training and Experimentation

Generate a PyTorch training script that:
1. Defines the InfoNCE loss for CLIP (contrastive loss over image-text pairs).
   - Use cosine similarity between image and text embeddings.
   - Include temperature parameter τ (learnable or fixed).
2. Trains the model to align image and text embeddings:
   - Supports configurable batch size, learning rate, and optimizer.
   - Use AdamW optimizer with weight decay.
3. Includes a training loop that logs training and validation losses each epoch.
4. Plots loss curves using matplotlib and saves them as "loss_curves.png".
5. Records:
   - Total training time
   - Hardware (GPU name)
6. Adds exception handling for potential issues (instability/divergence).

Include sensible defaults:
- lr = 1e-4
- batch_size = 64
- epochs = 10
- optimizer = AdamW



# ELEC 475 Lab 4 - Section 2.4 Evaluation and Visualization

Generate code to evaluate the fine-tuned CLIP model:
1. Compute cosine similarity between all validation image and text embeddings.
2. Implement Recall@K (K=1,5,10) for:
   - Image→Text retrieval
   - Text→Image retrieval
3. Print and save Recall@K results to a CSV or text file.
4. Visualize qualitative results:
   - Given a text query (e.g. "a dog playing"), show top-5 retrieved images.
   - Given an image and a list of classes (e.g. ["a person", "an animal", "a landscape"]), return the best-matching text.
5. Plot similarity heatmaps or example retrievals for the lab report.

Include comments explaining:
- How cosine similarity is computed
- What Recall@K measures and why it’s used


# ELEC 475 Lab 4 - Section 2.5 Modifications and Ablation Study

Generate an updated training script that applies at least TWO modifications to improve CLIP fine-tuning accuracy.
Possible modifications include:
- Adding dropout or layer normalization in the projection head
- Applying data augmentation (random crop, flip, color jitter)
- Using gradient clipping or learning rate warmup
- Unfreezing last few layers of the image encoder

Requirements:
1. Clearly separate baseline and modified model code.
2. Retrain the modified versions and evaluate using Recall@1,5,10.
3. Generate a short summary (in Markdown or text) comparing results across versions.
4. Plot a bar chart showing Recall@K improvement over the baseline.


# ELEC 475 Lab 4 - Section 2.6 Analysis and Discussion

Generate a concise Markdown summary (for inclusion in the lab report) that includes:
1. All hyperparameters used (batch size, LR, optimizer, epochs).
2. Hardware details (GPU type, runtime).
3. Description of CLIP loss intuition:
   - Anchors, positives, negatives, and temperature scaling.
4. Sample qualitative retrievals demonstrating semantic alignment.
5. Observations about training stability, convergence, and retrieval quality.
6. Reflection on LLM usage:
   - Which prompts were most useful?
   - How were outputs validated?
   - Include placeholder section for a link to conversation.
