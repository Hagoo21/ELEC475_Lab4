"""
Simple EDA script for COCO 2014 dataset
Analyzes train and validation images in the coco2014 folder
"""

import os
import json
from pathlib import Path
from collections import Counter
from statistics import mean, median
from PIL import Image
import sys

# Add project root to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_CAPTIONS_PATH, VAL_CAPTIONS_PATH


def count_images(directory):
    """Count number of image files in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    for file in Path(directory).iterdir():
        if file.suffix.lower() in image_extensions:
            count += 1
    return count


def get_image_stats(directory, max_samples=None):
    """Get statistics about images (dimensions, file sizes)"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    widths = []
    heights = []
    file_sizes = []
    aspect_ratios = []
    
    image_files = [f for f in Path(directory).iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if max_samples:
        import random
        random.seed(42)
        image_files = random.sample(image_files, min(max_samples, len(image_files)))
    
    print(f"  Analyzing {len(image_files)} images...")
    
    for img_file in image_files:
        try:
            # Get file size
            file_size = img_file.stat().st_size / (1024 * 1024)  # MB
            file_sizes.append(file_size)
            
            # Get image dimensions
            with Image.open(img_file) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
        except Exception as e:
            print(f"    Warning: Could not process {img_file.name}: {e}")
            continue
    
    stats = {
        'count': len(widths),
        'widths': widths,
        'heights': heights,
        'file_sizes': file_sizes,
        'aspect_ratios': aspect_ratios
    }
    return stats


def analyze_captions(captions_path):
    """Analyze caption statistics"""
    if not os.path.exists(captions_path):
        return None
    
    with open(captions_path, 'r') as f:
        data = json.load(f)
    
    captions = [ann['caption'] for ann in data.get('annotations', [])]
    
    # Count captions per image
    image_caption_count = Counter()
    for ann in data.get('annotations', []):
        image_caption_count[ann['image_id']] += 1
    
    # Word statistics
    all_words = []
    caption_lengths = []
    for caption in captions:
        words = caption.lower().split()
        all_words.extend(words)
        caption_lengths.append(len(words))
    
    word_freq = Counter(all_words)
    
    return {
        'total_captions': len(captions),
        'unique_images': len(set(ann['image_id'] for ann in data.get('annotations', []))),
        'captions_per_image': dict(image_caption_count),
        'avg_caption_length': mean(caption_lengths) if caption_lengths else 0,
        'top_words': word_freq.most_common(10),
        'caption_lengths': caption_lengths
    }


def print_stats(stats, split_name):
    """Print formatted statistics"""
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} SET STATISTICS")
    print(f"{'='*60}")
    
    if stats['count'] == 0:
        print("  No images found!")
        return
    
    print(f"\nImage Count: {stats['count']}")
    
    print(f"\nImage Dimensions:")
    print(f"  Width:  min={min(stats['widths']):.0f}, "
          f"max={max(stats['widths']):.0f}, "
          f"mean={mean(stats['widths']):.1f}, "
          f"median={median(stats['widths']):.1f}")
    print(f"  Height: min={min(stats['heights']):.0f}, "
          f"max={max(stats['heights']):.0f}, "
          f"mean={mean(stats['heights']):.1f}, "
          f"median={median(stats['heights']):.1f}")
    
    print(f"\nAspect Ratios:")
    print(f"  min={min(stats['aspect_ratios']):.2f}, "
          f"max={max(stats['aspect_ratios']):.2f}, "
          f"mean={mean(stats['aspect_ratios']):.2f}, "
          f"median={median(stats['aspect_ratios']):.2f}")
    
    print(f"\nFile Sizes:")
    print(f"  Total: {sum(stats['file_sizes']):.2f} MB")
    print(f"  Per image: min={min(stats['file_sizes']):.2f} MB, "
          f"max={max(stats['file_sizes']):.2f} MB, "
          f"mean={mean(stats['file_sizes']):.2f} MB, "
          f"median={median(stats['file_sizes']):.2f} MB")


def print_caption_stats(caption_stats, split_name):
    """Print caption statistics"""
    if caption_stats is None:
        print(f"\n  Caption file not found for {split_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} CAPTION STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nTotal Captions: {caption_stats['total_captions']}")
    print(f"Unique Images with Captions: {caption_stats['unique_images']}")
    print(f"Average Captions per Image: {caption_stats['total_captions'] / caption_stats['unique_images']:.2f}")
    print(f"Average Caption Length: {caption_stats['avg_caption_length']:.1f} words")
    
    caption_lengths = caption_stats['caption_lengths']
    if caption_lengths:
        print(f"Caption Length Range: {min(caption_lengths)} - {max(caption_lengths)} words")
    
    print(f"\nTop 10 Most Common Words:")
    for word, count in caption_stats['top_words']:
        print(f"  '{word}': {count}")


def main():
    """Main EDA function"""
    print("="*60)
    print("COCO 2014 Dataset - Exploratory Data Analysis")
    print("="*60)
    
    # Check if directories exist
    if not os.path.exists(TRAIN_IMAGES_DIR):
        print(f"Error: Train images directory not found: {TRAIN_IMAGES_DIR}")
        return
    
    if not os.path.exists(VAL_IMAGES_DIR):
        print(f"Error: Validation images directory not found: {VAL_IMAGES_DIR}")
        return
    
    # Count images
    print("\nCounting images...")
    train_count = count_images(TRAIN_IMAGES_DIR)
    val_count = count_images(VAL_IMAGES_DIR)
    
    print(f"\nTotal Images Found:")
    print(f"  Train: {train_count}")
    print(f"  Validation: {val_count}")
    print(f"  Total: {train_count + val_count}")
    
    # Analyze train images (sample if too many)
    print(f"\nAnalyzing train images...")
    max_samples = 1000 if train_count > 1000 else None
    train_stats = get_image_stats(TRAIN_IMAGES_DIR, max_samples=max_samples)
    print_stats(train_stats, "Train")
    
    # Analyze validation images (sample if too many)
    print(f"\nAnalyzing validation images...")
    max_samples = 1000 if val_count > 1000 else None
    val_stats = get_image_stats(VAL_IMAGES_DIR, max_samples=max_samples)
    print_stats(val_stats, "Validation")
    
    # Analyze captions if available
    print(f"\nAnalyzing captions...")
    train_caption_stats = analyze_captions(TRAIN_CAPTIONS_PATH)
    print_caption_stats(train_caption_stats, "Train")
    
    val_caption_stats = analyze_captions(VAL_CAPTIONS_PATH)
    print_caption_stats(val_caption_stats, "Validation")
    
    print(f"\n{'='*60}")
    print("EDA Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

