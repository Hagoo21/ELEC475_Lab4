# ELEC475_Lab4


# ELEC 475 Lab 4 - Section 2.1 Dataset Preparation (MS COCO)
# Goal: Prepare the COCO 2014 dataset for CLIP fine-tuning.

Generate a Python script using PyTorch that:
1. Loads the COCO 2014 dataset:
   - train2014/ and val2014/ image folders
   - captions_train2014.json and captions_val2014.json
2. Implements preprocessing for both modalities:
   - Resize all images to 224x224
   - Normalize using CLIP mean=[0.48145466, 0.4578275, 0.40821073] and std=[0.26862954, 0.26130258, 0.27577711]
3. Uses the pretrained CLIP text encoder ("openai/clip-vit-base-patch32" from HuggingFace) to encode captions.
4. Creates a PyTorch dataset class that returns:
   - normalized image tensor
   - text embedding (from cached .pt files)
   - image_id
5. Adds functionality to cache all text embeddings into `train_text_embeds.pt` and `val_text_embeds.pt`
6. Displays a few random image-caption pairs to verify dataset integrity.

After generating the code, include short documentation comments explaining:
- Dataset split sizes used (full or subset)
- Image preprocessing pipeline
- Text preprocessing (tokenization, max length, padding)


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
