import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPModel

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Configuration  ---
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 224

# Paths (Adjust to your local structure)
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "coco2014")  # Points to data/coco2014/
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images", "train2014")
VAL_IMG_DIR = os.path.join(DATA_DIR, "images", "val2014")
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
    Encodes captions using pretrained CLIP model and saves to .pt file.
    Uses CLIPModel (not CLIPTextModel) to ensure text_projection is applied,
    matching the embeddings used during training.
    This fulfills the hint to cache embeddings[cite: 57].
    """
    print(f"Processing captions from {json_path}...")
    
    # Load Pretrained CLIP Model (includes text_projection layer)
    # Using CLIPModel instead of CLIPTextModel to match training code
    # Force fresh download to avoid corrupted cache issues
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, 
        force_download=True, 
        resume_download=False,
        local_files_only=False
    )
    clip_model = CLIPModel.from_pretrained(
        model_id,
        force_download=True,
        resume_download=False,
        local_files_only=False
    ).to(device)
    clip_model.eval()

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

            # Encode using CLIPModel (includes text_projection)
            # This matches the process in models/clip_model.py
            text_outputs = clip_model.text_model(**inputs)
            text_embeds = text_outputs.pooler_output
            
            # Apply text_projection (same as in CLIPTextEncoder.forward)
            if hasattr(clip_model, 'text_projection') and clip_model.text_projection is not None:
                text_embeds = clip_model.text_projection(text_embeds)
            
            # L2 normalize embeddings (same as in training)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
            
            # Move to CPU for storage
            text_embeds = text_embeds.cpu()

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

def main():
    """
    Main function to cache text embeddings for train and validation sets.
    """
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(SCRIPT_DIR, "..", "datasets", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file paths
    train_cache_path = os.path.join(cache_dir, "train_text_embeds.pt")
    val_cache_path = os.path.join(cache_dir, "val_text_embeds.pt")
    
    # Check if files already exist
    if os.path.exists(train_cache_path):
        print(f"⚠️  Train cache already exists at {train_cache_path}")
        response = input("Do you want to regenerate it? (y/n): ")
        if response.lower() != 'y':
            print("Skipping train cache generation.")
        else:
            cache_text_embeddings(TRAIN_CAP_FILE, train_cache_path)
    else:
        cache_text_embeddings(TRAIN_CAP_FILE, train_cache_path)
    
    if os.path.exists(val_cache_path):
        print(f"⚠️  Val cache already exists at {val_cache_path}")
        response = input("Do you want to regenerate it? (y/n): ")
        if response.lower() != 'y':
            print("Skipping val cache generation.")
        else:
            cache_text_embeddings(VAL_CAP_FILE, val_cache_path)
    else:
        cache_text_embeddings(VAL_CAP_FILE, val_cache_path)
    
    print("\n✅ Preprocessing complete!")

if __name__ == "__main__":
    main()