import os
import json
import torch
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Configuration  ---
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 224

# Paths (Adjust to your local structure)
DATA_DIR = "./coco2014" # Example root
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train2014")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val2014")
TRAIN_CAP_FILE = os.path.join(DATA_DIR, "annotations/captions_train2014.json")
VAL_CAP_FILE = os.path.join(DATA_DIR, "annotations/captions_val2014.json")

def get_transforms():
    """
    Returns the image preprocessing pipeline required by CLIP.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize to 224x224 
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD) # Normalize [cite: 53, 56]
    ])

def cache_text_embeddings(json_path, save_path, batch_size=32):
    """
    Encodes captions using pretrained CLIP text encoder and saves to .pt file.
    This fulfills the hint to cache embeddings[cite: 57].
    """
    print(f"Processing captions from {json_path}...")
    
    # Load Pretrained CLIP Text Encoder 
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = CLIPTextModel.from_pretrained(model_id).to(device)
    text_encoder.eval()

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    # Create a mapping of image_id -> filename if needed, 
    # but typically we store list of (image_id, caption)
    
    cached_data = []
    
    # Process in batches for efficiency
    with torch.no_grad():
        for i in range(0, len(annotations), batch_size):
            batch = annotations[i:i+batch_size]
            captions = [item['caption'] for item in batch]
            image_ids = [item['image_id'] for item in batch]
            ids = [item['id'] for item in batch] # Caption IDs

            # Tokenize
            inputs = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt", max_length=77
            ).to(device)

            # Encode
            outputs = text_encoder(**inputs)
            text_embeds = outputs.pooler_output.cpu() # Move back to CPU for storage

            for j, embed in enumerate(text_embeds):
                cached_data.append({
                    "image_id": image_ids[j],
                    "caption_id": ids[j],
                    "caption_text": captions[j],
                    "text_embedding": embed
                })
            
            if i % 1000 == 0:
                print(f"Processed {i}/{len(annotations)} captions")

    print(f"Saving cache to {save_path}...")
    torch.save(cached_data, save_path)
    print("Done.")

# --- Run Caching (Uncomment to run once) ---
# cache_text_embeddings(TRAIN_CAP_FILE, "train_cache.pt")
# cache_text_embeddings(VAL_CAP_FILE, "val_cache.pt")