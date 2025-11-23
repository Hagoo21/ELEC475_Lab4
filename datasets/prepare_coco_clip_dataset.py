"""
COCO-CLIP Dataset Preparation Script

This script downloads and prepares the COCO 2014 dataset (or a subset) for use with CLIP-style models.
It handles dataset downloading, sampling, preprocessing, text encoding, and caching.

Dataset Details:
    - Train split: Configurable (set TRAIN_SAMPLES in config.py, None for entire dataset)
    - Val split: Configurable (set VAL_SAMPLES in config.py, None for entire dataset)
    - Random seed: 42 (for reproducibility)
    - Full dataset sizes: ~82,783 train images (~13GB), ~40,504 val images (~6GB)

Image Preprocessing Pipeline:
    1. Resize to 224x224 pixels
    2. Convert to tensor
    3. Normalize using CLIP statistics:
       - mean = [0.48145466, 0.4578275, 0.40821073]
       - std = [0.26862954, 0.26130258, 0.27577711]

Text Preprocessing:
    - Tokenizer: CLIPTokenizer from "openai/clip-vit-base-patch32"
    - max_length: 77 (CLIP's default context length)
    - truncation: True (truncate sequences longer than max_length)
    - padding: 'max_length' (pad all sequences to max_length)
    - return_tensors: 'pt' (PyTorch tensors)

Caching Strategy:
    Text embeddings are precomputed and cached to disk because:
    1. Speed: Avoids recomputing embeddings during every training epoch
    2. GPU Memory: Embeddings can be loaded directly without loading the CLIP text encoder during training
    3. Efficiency: Text encoder forward pass is expensive; doing it once saves significant computation
    
    Cache files:
        - datasets/cache/train_text_embeds.pt (contains precomputed training caption embeddings)
        - datasets/cache/val_text_embeds.pt (contains precomputed validation caption embeddings)
"""

import os
import sys
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
# Allow loading truncated images (some COCO images may be corrupted/truncated)
ImageFile.LOAD_TRUNCATED_IMAGES = True

import fiftyone as fo
import fiftyone.zoo as foz
from transformers import CLIPTokenizer, CLIPTextModel

import matplotlib.pyplot as plt

# Import global configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set random seeds for reproducibility
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)


# Get the directory where this script is located (datasets folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cleanup_fiftyone_cache():
    """
    Clean up FiftyOne's cached downloads to fix corrupted/incomplete downloads.
    This removes the FiftyOne database and zoo dataset cache.
    """
    print("\n" + "="*60)
    print("Cleaning up FiftyOne cache to fix corrupted downloads...")
    print("="*60)
    
    try:
        # Delete any temporary datasets
        print("Deleting temporary FiftyOne datasets...")
        for dataset_name in fo.list_datasets():
            if 'coco-2014' in dataset_name.lower() and 'temp' in dataset_name.lower():
                try:
                    fo.delete_dataset(dataset_name)
                    print(f"  ✓ Deleted dataset: {dataset_name}")
                except Exception as e:
                    print(f"  ⚠ Could not delete {dataset_name}: {e}")
        
        # Get FiftyOne paths
        try:
            # Try to get the zoo dataset directory
            if hasattr(fo.config, 'dataset_zoo_dir'):
                zoo_dir = fo.config.dataset_zoo_dir
            elif hasattr(fo.config, 'default_dataset_dir'):
                zoo_dir = fo.config.default_dataset_dir
            else:
                # Default location
                from pathlib import Path
                zoo_dir = str(Path.home() / 'fiftyone')
            
            # Clean up COCO-specific cache
            if os.path.exists(zoo_dir):
                coco_cache = os.path.join(zoo_dir, 'coco-2014')
                if os.path.exists(coco_cache):
                    import shutil
                    print(f"\nRemoving COCO cache at: {coco_cache}")
                    shutil.rmtree(coco_cache)
                    print("  ✓ Removed COCO cache")
        except Exception as e:
            print(f"  ⚠ Could not clean cache directory: {e}")
            print("  You may need to manually delete the FiftyOne cache")
            print("  Default location: ~/fiftyone or C:\\Users\\<username>\\fiftyone")
        
        print("\n✓ Cleanup complete. You can now retry the download.")
        
    except Exception as e:
        print(f"⚠ Error during cleanup: {e}")
        print("\nManual cleanup instructions:")
        print("1. Delete temporary FiftyOne datasets:")
        print("   - Run: python -c \"import fiftyone as fo; [fo.delete_dataset(d) for d in fo.list_datasets() if 'coco-2014' in d.lower()]\"")
        print("2. Delete FiftyOne cache directory:")
        print("   - Windows: C:\\Users\\<username>\\fiftyone")
        print("   - Linux/Mac: ~/fiftyone")


def _download_coco_captions(split: str) -> str:
    """
    Download COCO caption annotations if not already cached.
    
    Args:
        split: Dataset split ('train' or 'val')
        
    Returns:
        Path to the downloaded caption annotation file
    """
    import urllib.request
    import zipfile
    
    # COCO caption annotation URLs
    caption_urls = {
        'train': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'val': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    }
    
    # Map split to annotation filename
    annotation_files = {
        'train': 'captions_train2014.json',
        'val': 'captions_val2014.json'
    }
    
    # Create cache directory
    cache_dir = config.ANNOTATIONS_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    
    annotation_file = annotation_files[split]
    annotation_path = os.path.join(cache_dir, annotation_file)
    
    # Check if already downloaded
    if os.path.exists(annotation_path):
        print(f"✓ Caption annotations already cached at {annotation_path}")
        return annotation_path
    
    # Download and extract annotations
    zip_path = os.path.join(cache_dir, 'annotations_trainval2014.zip')
    
    if not os.path.exists(zip_path):
        print(f"Downloading caption annotations (~19MB)...")
        url = caption_urls[split]
        urllib.request.urlretrieve(url, zip_path)
        print("✓ Download complete")
    
    # Extract the specific annotation file we need
    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract only the caption file we need
        zip_ref.extract(f'annotations/{annotation_file}', cache_dir)
    
    # Move to cache_dir root
    import shutil
    src = os.path.join(cache_dir, 'annotations', annotation_file)
    shutil.move(src, annotation_path)
    
    # Clean up
    os.rmdir(os.path.join(cache_dir, 'annotations'))
    
    # Clean up zip file after successful extraction (optional - saves 19MB)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("✓ Cleaned up zip file")
    
    print(f"✓ Annotations extracted to {annotation_path}")
    return annotation_path


def download_and_sample_coco(
    split: str,
    num_samples: Optional[int],
    output_dir: str,
    max_retries: int = 3
) -> Tuple[List[str], Dict]:
    """
    Download COCO 2014 dataset, sample images, and export to local directory.
    
    If num_samples is None, downloads the entire dataset split.
    If num_samples is an integer, downloads only that many images.
    
    Args:
        split: Dataset split ('train' or 'val')
        num_samples: Number of images to sample (None for full dataset)
        output_dir: Directory to export images and captions
        max_retries: Maximum number of retry attempts if download fails
        
    Returns:
        Tuple of (image_paths, captions_dict)
    """
    print(f"\n{'='*60}")
    print(f"Downloading COCO 2014 {split} split...")
    if num_samples is None:
        print(f"Note: Downloading ENTIRE dataset (this may take a while and require ~13GB for train, ~6GB for val)")
    else:
        print(f"Note: Downloading {num_samples} images (subset of full dataset)")
    print(f"{'='*60}")
    
    # Map 'val' to 'validation' for FiftyOne
    fiftyone_split = 'validation' if split == 'val' else split
    
    # Retry logic for handling corrupted downloads
    for attempt in range(1, max_retries + 1):
        try:
            # Load COCO dataset using FiftyOne (images only, captions loaded separately)
            # If num_samples is None, don't use max_samples to download the entire dataset
            # Note: We don't specify label_types because COCO 2014 doesn't support "captions" type in FiftyOne
            if num_samples is None:
                # Download entire dataset
                dataset = foz.load_zoo_dataset(
                    config.COCO_DATASET_NAME,
                    split=fiftyone_split,
                    shuffle=True,
                    seed=config.RANDOM_SEED,
                    dataset_name=f"coco-2014-{split}-temp"
                )
            else:
                # Download subset
                dataset = foz.load_zoo_dataset(
                    config.COCO_DATASET_NAME,
                    split=fiftyone_split,
                    max_samples=num_samples,
                    shuffle=True,
                    seed=config.RANDOM_SEED,
                    dataset_name=f"coco-2014-{split}-temp"
                )
            
            print(f"Downloaded {len(dataset)} images")
            break  # Success, exit retry loop
            
        except (EOFError, zipfile.BadZipFile, OSError) as e:
            if attempt < max_retries:
                print(f"\n⚠ Download attempt {attempt} failed with error: {e}")
                print(f"Cleaning up corrupted cache and retrying ({attempt}/{max_retries})...")
                cleanup_fiftyone_cache()
                print("Retrying download...\n")
            else:
                print(f"\n❌ Download failed after {max_retries} attempts.")
                print(f"Error: {e}")
                print("\nPossible solutions:")
                print("1. Check your internet connection")
                print("2. Ensure you have enough disk space (~20GB for full dataset)")
                print("3. Manually clean FiftyOne cache:")
                print("   - Windows: Delete C:\\Users\\<username>\\fiftyone")
                print("   - Linux/Mac: Delete ~/fiftyone")
                print("4. Try running the script again")
                raise
        except Exception as e:
            # For other errors, don't retry
            print(f"\n❌ Unexpected error during download: {e}")
            raise
    
    if 'dataset' not in locals():
        raise RuntimeError("Failed to download dataset after all retries")
    
    # Download caption annotations separately
    print("Downloading COCO caption annotations...")
    caption_ann_path = _download_coco_captions(split)
    
    # Load caption annotations
    with open(caption_ann_path, 'r') as f:
        caption_data = json.load(f)
    
    # Create mapping from image_id to captions
    image_id_to_captions = {}
    for ann in caption_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_captions:
            image_id_to_captions[img_id] = []
        image_id_to_captions[img_id].append(ann['caption'])
    
    # Use the downloaded samples directly (already limited to num_samples)
    sampled_view = dataset
    
    print(f"Sampled {len(sampled_view)} images")
    
    # Create output directories
    images_dir = os.path.join(output_dir, split, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Prepare COCO caption format
    coco_captions = {
        'images': [],
        'annotations': []
    }
    
    annotation_id = 0
    image_paths = []
    
    # Export images and prepare captions
    for idx, sample in enumerate(sampled_view):
        # Get image info
        image_id = int(sample.filepath.split('_')[-1].split('.')[0])  # Extract from filename
        filename = os.path.basename(sample.filepath)
        
        # Copy image to output directory
        from shutil import copy2
        dest_path = os.path.join(images_dir, filename)
        copy2(sample.filepath, dest_path)
        image_paths.append(dest_path)
        
        # Add image info
        img = Image.open(dest_path)
        width, height = img.size
        coco_captions['images'].append({
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height
        })
        
        # Add caption annotations from the downloaded annotation file
        if image_id in image_id_to_captions:
            for caption in image_id_to_captions[image_id]:
                coco_captions['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'caption': caption
                })
                annotation_id += 1
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(sampled_view)} images")
    
    # Save captions in COCO format
    captions_path = os.path.join(output_dir, split, 'captions.json')
    with open(captions_path, 'w') as f:
        json.dump(coco_captions, f, indent=2)
    
    print(f"✓ Exported {len(image_paths)} images to {images_dir}")
    print(f"✓ Saved captions to {captions_path}")
    
    # Clean up FiftyOne dataset
    fo.delete_dataset(f"coco-2014-{split}-temp")
    
    return image_paths, coco_captions


def load_captions_from_json(captions_path: str) -> Dict[int, List[str]]:
    """
    Load captions from COCO format JSON file.
    
    Args:
        captions_path: Path to captions.json
        
    Returns:
        Dictionary mapping image_id to list of captions
    """
    with open(captions_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from image_id to captions
    image_to_captions = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        
        if image_id not in image_to_captions:
            image_to_captions[image_id] = []
        image_to_captions[image_id].append(caption)
    
    return image_to_captions


def precompute_text_embeddings(
    captions_path: str,
    cache_path: str,
    model_name: str = None,
    max_length: int = None,
    device: str = None
) -> Dict[int, torch.Tensor]:
    """
    Precompute and cache CLIP text embeddings for all captions.
    
    Args:
        captions_path: Path to captions.json
        cache_path: Path to save cached embeddings
        model_name: HuggingFace CLIP model name (defaults to config value)
        max_length: Maximum sequence length for tokenization (defaults to config value)
        device: Device to run model on (defaults to config value or auto-detect)
        
    Returns:
        Dictionary mapping image_id to text embedding tensor
    """
    # Use config defaults if not specified
    if model_name is None:
        model_name = config.CLIP_MODEL_NAME
    if max_length is None:
        max_length = config.CLIP_MAX_LENGTH
    if device is None:
        device = config.DEVICE if torch.cuda.is_available() else 'cpu'
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading cached text embeddings from {cache_path}")
        return torch.load(cache_path)
    
    print(f"\nPrecomputing text embeddings using {model_name}...")
    print(f"Device: {device}")
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
    text_encoder.eval()
    
    # Load captions
    image_to_captions = load_captions_from_json(captions_path)
    
    # Precompute embeddings
    image_to_embeddings = {}
    
    with torch.no_grad():
        for idx, (image_id, captions) in enumerate(image_to_captions.items()):
            # Use the first caption (or you could average multiple captions)
            caption = captions[0]
            
            # Tokenize caption
            # Tokenization strategy:
            # - max_length=77: CLIP's context length
            # - truncation=True: Truncate sequences longer than max_length
            # - padding='max_length': Pad all sequences to max_length for uniform tensor size
            # - return_tensors='pt': Return PyTorch tensors
            inputs = tokenizer(
                caption,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Encode text
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get pooled output (last hidden state of [EOS] token)
            text_embedding = outputs.pooler_output  # Shape: (1, 512)
            
            # Store embedding (on CPU to save GPU memory)
            image_to_embeddings[image_id] = text_embedding.cpu().squeeze(0)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_to_captions)} captions")
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(image_to_embeddings, cache_path)
    print(f"✓ Saved text embeddings to {cache_path}")
    
    return image_to_embeddings


class COCOCLIPDataset(Dataset):
    """
    PyTorch Dataset for COCO images with precomputed CLIP text embeddings.
    
    Returns:
        image: Normalized image tensor (3, 224, 224)
        text_embedding: Precomputed CLIP text embedding (512,)
        image_id: Image ID from COCO dataset
    """
    
    def __init__(
        self,
        images_dir: str,
        captions_path: str,
        text_embeddings: Dict[int, torch.Tensor],
        transform=None
    ):
        """
        Args:
            images_dir: Directory containing images
            captions_path: Path to captions.json
            text_embeddings: Dictionary mapping image_id to text embeddings
            transform: Optional image transforms
        """
        self.images_dir = images_dir
        self.text_embeddings = text_embeddings
        
        # Load image metadata
        with open(captions_path, 'r') as f:
            coco_data = json.load(f)
        self.images = coco_data['images']
        
        # Load captions for verification
        self.image_to_captions = load_captions_from_json(captions_path)
        
        # Set up transforms
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform
    
    def _default_transform(self):
        """
        Create default CLIP-style image preprocessing transform.
        
        Pipeline:
            1. Resize to 224x224
            2. Convert to tensor (scales to [0, 1])
            3. Normalize with CLIP statistics
        """
        return transforms.Compose([
            transforms.Resize((config.CLIP_IMAGE_SIZE, config.CLIP_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.CLIP_MEAN, std=config.CLIP_STD)
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        image_info = self.images[idx]
        image_id = image_info['id']
        filename = image_info['file_name']
        
        # Load and transform image
        image_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except (OSError, IOError) as e:
            # If image is corrupted and can't be loaded, create a black image as fallback
            print(f"Warning: Could not load image {filename} (ID: {image_id}): {e}")
            # Create a black image of the expected size (224x224 for CLIP)
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            image_tensor = self.transform(image)
        
        # Get precomputed text embedding
        text_embedding = self.text_embeddings[image_id]
        
        return image_tensor, text_embedding, image_id
    
    def get_raw_caption(self, image_id: int) -> str:
        """Get the raw caption for an image (before tokenization)."""
        captions = self.image_to_captions.get(image_id, ["No caption available"])
        return captions[0]
    
    def get_image_path(self, idx: int) -> str:
        """Get the file path for an image."""
        image_info = self.images[idx]
        filename = image_info['file_name']
        return os.path.join(self.images_dir, filename)


def verify_dataset(dataset: COCOCLIPDataset, num_samples: int = 5):
    """
    Verify dataset by displaying random samples with images and captions.
    
    Args:
        dataset: COCOCLIPDataset instance
        num_samples: Number of samples to display
    """
    print(f"\n{'='*60}")
    print(f"Dataset Verification - Displaying {num_samples} Random Samples")
    print(f"{'='*60}")
    
    # Randomly sample indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get sample
        image_tensor, text_embedding, image_id = dataset[idx]
        
        # Get raw caption
        raw_caption = dataset.get_raw_caption(image_id)
        
        # Denormalize image for display
        mean = torch.tensor(config.CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(config.CLIP_STD).view(3, 1, 1)
        image_denorm = image_tensor * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        # Convert to numpy for display
        image_np = image_denorm.permute(1, 2, 0).numpy()
        
        # Display image
        axes[i].imshow(image_np)
        axes[i].axis('off')
        axes[i].set_title(f"Image ID: {image_id}\n{raw_caption[:50]}...", 
                          fontsize=9, wrap=True)
        
        # Print detailed info
        print(f"\nSample {i+1}:")
        print(f"  Image ID: {image_id}")
        print(f"  Image tensor shape: {image_tensor.shape}")
        print(f"  Text embedding shape: {text_embedding.shape}")
        print(f"  Raw caption: {raw_caption}")
    
    plt.tight_layout()
    
    # Save figure in the datasets directory (same location as script)
    output_path = os.path.join(SCRIPT_DIR, 'verification_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved verification figure to {output_path}")
    plt.close()
    
    # Print tensor shape summary
    print(f"\n{'='*60}")
    print("Tensor Shape Summary:")
    print(f"{'='*60}")
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"  - Channels: {image_tensor.shape[0]}")
    print(f"  - Height: {image_tensor.shape[1]}")
    print(f"  - Width: {image_tensor.shape[2]}")
    print(f"\nText embedding shape: {text_embedding.shape}")
    print(f"  - Embedding dimension: {text_embedding.shape[0]}")


def main():
    """
    Main function to prepare COCO-CLIP dataset end-to-end.
    """
    print("="*60)
    print("COCO-CLIP Dataset Preparation")
    print("="*60)
    print(f"Script location: {SCRIPT_DIR}")
    print(f"\nConfiguration:")
    train_samples_str = "ALL (entire dataset)" if config.TRAIN_SAMPLES is None else str(config.TRAIN_SAMPLES)
    val_samples_str = "ALL (entire dataset)" if config.VAL_SAMPLES is None else str(config.VAL_SAMPLES)
    print(f"  Train samples: {train_samples_str}")
    print(f"  Val samples: {val_samples_str}")
    print(f"  Output directory: {config.DATA_DIR}")
    print(f"  Cache directory: {config.CACHE_DIR}")
    print(f"  Random seed: {config.RANDOM_SEED}")
    print(f"  CLIP model: {config.CLIP_MODEL_NAME}")
    
    # Step 1: Download and sample COCO dataset
    # Check if data already exists
    if not os.path.exists(config.TRAIN_CAPTIONS_PATH):
        download_and_sample_coco(
            'train',
            config.TRAIN_SAMPLES,
            config.DATA_DIR
        )
    else:
        print(f"\n✓ Train split already exists at {config.TRAIN_IMAGES_DIR}")
    
    if not os.path.exists(config.VAL_CAPTIONS_PATH):
        download_and_sample_coco(
            'val',
            config.VAL_SAMPLES,
            config.DATA_DIR
        )
    else:
        print(f"✓ Val split already exists at {config.VAL_IMAGES_DIR}")
    
    # Step 2 & 3: Precompute text embeddings with caching
    train_text_embeddings = precompute_text_embeddings(
        config.TRAIN_CAPTIONS_PATH,
        config.TRAIN_EMBEDDINGS_PATH
    )
    
    val_text_embeddings = precompute_text_embeddings(
        config.VAL_CAPTIONS_PATH,
        config.VAL_EMBEDDINGS_PATH
    )
    
    # Step 4: Create PyTorch datasets
    print(f"\n{'='*60}")
    print("Creating PyTorch datasets...")
    print(f"{'='*60}")
    
    train_dataset = COCOCLIPDataset(
        config.TRAIN_IMAGES_DIR,
        config.TRAIN_CAPTIONS_PATH,
        train_text_embeddings
    )
    
    val_dataset = COCOCLIPDataset(
        config.VAL_IMAGES_DIR,
        config.VAL_CAPTIONS_PATH,
        val_text_embeddings
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")
    
    # Step 5: Verify dataset
    verify_dataset(train_dataset, num_samples=5)
    
    # Step 6: Create sample dataloaders to demonstrate usage
    print(f"\n{'='*60}")
    print("Creating sample DataLoaders...")
    print(f"{'='*60}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    batch = next(iter(train_loader))
    images, text_embeds, image_ids = batch
    print(f"  Batch image shape: {images.shape}")
    print(f"  Batch text embedding shape: {text_embeds.shape}")
    print(f"  Batch image IDs: {image_ids[:5].tolist()}...")
    
    print(f"\n{'='*60}")
    print("✓ Dataset preparation complete!")
    print(f"{'='*60}")
    print("\nDataset is ready for training. You can now:")
    print("  1. Import COCOCLIPDataset from this module")
    print("  2. Create train/val datasets with cached embeddings")
    print("  3. Use with PyTorch DataLoader for training")
    print(f"\nCache files:")
    print(f"  - {config.TRAIN_EMBEDDINGS_PATH}")
    print(f"  - {config.VAL_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare COCO-CLIP dataset')
    parser.add_argument(
        '--cleanup-cache',
        action='store_true',
        help='Clean up FiftyOne cache to fix corrupted downloads (then exit)'
    )
    args = parser.parse_args()
    
    if args.cleanup_cache:
        print("Cleaning up FiftyOne cache...")
        cleanup_fiftyone_cache()
        print("\n✓ Cleanup complete. You can now run the script normally.")
    else:
        main()

