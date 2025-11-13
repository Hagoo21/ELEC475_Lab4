"""
Exploratory Data Analysis (EDA) for COCO-CLIP Dataset

This script provides a comprehensive analysis of the downloaded COCO dataset,
including statistics, visualizations, and data quality checks.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
TRAIN_IMAGES_DIR = os.path.join(SCRIPT_DIR, 'coco_subset', 'train', 'images')
VAL_IMAGES_DIR = os.path.join(SCRIPT_DIR, 'coco_subset', 'val', 'images')
TRAIN_CAPTIONS_PATH = os.path.join(SCRIPT_DIR, 'coco_subset', 'train', 'captions.json')
VAL_CAPTIONS_PATH = os.path.join(SCRIPT_DIR, 'coco_subset', 'val', 'captions.json')
TRAIN_EMBEDS_PATH = os.path.join(SCRIPT_DIR, 'cache', 'train_text_embeds.pt')
VAL_EMBEDS_PATH = os.path.join(SCRIPT_DIR, 'cache', 'val_text_embeds.pt')

# Output directory for EDA results
EDA_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'eda_results')
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)


def load_data(split='train'):
    """Load captions and embeddings for a given split."""
    if split == 'train':
        captions_path = TRAIN_CAPTIONS_PATH
        embeds_path = TRAIN_EMBEDS_PATH
    else:
        captions_path = VAL_CAPTIONS_PATH
        embeds_path = VAL_EMBEDS_PATH
    
    # Load captions
    with open(captions_path, 'r') as f:
        data = json.load(f)
    
    # Load embeddings
    embeddings = torch.load(embeds_path)
    
    return data, embeddings


def analyze_basic_stats(split='train'):
    """Analyze basic dataset statistics."""
    print(f"\n{'='*70}")
    print(f"BASIC STATISTICS - {split.upper()} SPLIT")
    print(f"{'='*70}")
    
    data, embeddings = load_data(split)
    
    num_images = len(data['images'])
    num_captions = len(data['annotations'])
    num_embeddings = len(embeddings)
    
    print(f"Number of images: {num_images:,}")
    print(f"Number of captions: {num_captions:,}")
    print(f"Number of embeddings: {num_embeddings:,}")
    print(f"Captions per image: {num_captions / num_images:.1f}")
    
    # Check embedding dimensions
    first_embed = next(iter(embeddings.values()))
    print(f"Embedding dimension: {first_embed.shape[0]}")
    print(f"Embedding dtype: {first_embed.dtype}")
    
    return data, embeddings


def analyze_image_properties(data, split='train'):
    """Analyze image dimensions and aspect ratios."""
    print(f"\n{'='*70}")
    print(f"IMAGE PROPERTIES - {split.upper()} SPLIT")
    print(f"{'='*70}")
    
    widths = [img['width'] for img in data['images']]
    heights = [img['height'] for img in data['images']]
    aspect_ratios = [w/h for w, h in zip(widths, heights)]
    
    print(f"Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}")
    print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}")
    print(f"Aspect Ratio - Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Mean: {np.mean(aspect_ratios):.2f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Width distribution
    axes[0, 0].hist(widths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Image Width Distribution')
    axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
    axes[0, 0].legend()
    
    # Height distribution
    axes[0, 1].hist(heights, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Image Height Distribution')
    axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
    axes[0, 1].legend()
    
    # Aspect ratio distribution
    axes[1, 0].hist(aspect_ratios, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Aspect Ratio (width/height)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].axvline(np.mean(aspect_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(aspect_ratios):.2f}')
    axes[1, 0].legend()
    
    # Scatter: Width vs Height
    axes[1, 1].scatter(widths, heights, alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Width (pixels)')
    axes[1, 1].set_ylabel('Height (pixels)')
    axes[1, 1].set_title('Width vs Height Scatter')
    axes[1, 1].plot([0, max(widths)], [0, max(widths)], 'r--', alpha=0.5, label='Square (1:1)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, f'{split}_image_properties.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved image properties plot to eda_results/{split}_image_properties.png")
    plt.close()


def analyze_caption_statistics(data, split='train'):
    """Analyze caption text statistics."""
    print(f"\n{'='*70}")
    print(f"CAPTION STATISTICS - {split.upper()} SPLIT")
    print(f"{'='*70}")
    
    captions = [ann['caption'] for ann in data['annotations']]
    
    # Length statistics (characters)
    caption_lengths = [len(cap) for cap in captions]
    print(f"Caption length (characters):")
    print(f"  Min: {min(caption_lengths)}, Max: {max(caption_lengths)}, Mean: {np.mean(caption_lengths):.1f}")
    
    # Word count statistics
    word_counts = [len(cap.split()) for cap in captions]
    print(f"\nCaption length (words):")
    print(f"  Min: {min(word_counts)}, Max: {max(word_counts)}, Mean: {np.mean(word_counts):.1f}")
    
    # Collect all words
    all_words = []
    for cap in captions:
        all_words.extend(cap.lower().split())
    
    print(f"\nVocabulary size: {len(set(all_words)):,} unique words")
    print(f"Total words: {len(all_words):,}")
    
    # Most common words
    word_freq = Counter(all_words)
    print(f"\nTop 20 most common words:")
    for word, count in word_freq.most_common(20):
        print(f"  '{word}': {count:,}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Caption length (characters)
    axes[0, 0].hist(caption_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Caption Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Caption Length Distribution (Characters)')
    axes[0, 0].axvline(np.mean(caption_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(caption_lengths):.1f}')
    axes[0, 0].legend()
    
    # Caption length (words)
    axes[0, 1].hist(word_counts, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Caption Length (words)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Caption Length Distribution (Words)')
    axes[0, 1].axvline(np.mean(word_counts), color='red', linestyle='--', label=f'Mean: {np.mean(word_counts):.1f}')
    axes[0, 1].legend()
    
    # Top words bar chart
    top_words = word_freq.most_common(15)
    words, counts = zip(*top_words)
    axes[1, 0].barh(range(len(words)), counts, color='lightgreen', alpha=0.7)
    axes[1, 0].set_yticks(range(len(words)))
    axes[1, 0].set_yticklabels(words)
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title('Top 15 Most Common Words')
    axes[1, 0].invert_yaxis()
    
    # Word frequency distribution (log scale)
    freq_values = sorted(word_freq.values(), reverse=True)
    axes[1, 1].plot(freq_values, linewidth=2)
    axes[1, 1].set_xlabel('Word Rank')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Word Frequency Distribution (Zipf\'s Law)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, f'{split}_caption_statistics.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved caption statistics plot to eda_results/{split}_caption_statistics.png")
    plt.close()


def analyze_embeddings(embeddings, split='train'):
    """Analyze CLIP text embedding properties."""
    print(f"\n{'='*70}")
    print(f"EMBEDDING ANALYSIS - {split.upper()} SPLIT")
    print(f"{'='*70}")
    
    # Convert to tensor
    embed_list = [embeddings[img_id] for img_id in sorted(embeddings.keys())]
    embed_tensor = torch.stack(embed_list)  # Shape: (N, 512)
    
    print(f"Embedding tensor shape: {embed_tensor.shape}")
    print(f"Memory size: {embed_tensor.element_size() * embed_tensor.nelement() / (1024**2):.2f} MB")
    
    # Statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embed_tensor.mean().item():.6f}")
    print(f"  Std: {embed_tensor.std().item():.6f}")
    print(f"  Min: {embed_tensor.min().item():.6f}")
    print(f"  Max: {embed_tensor.max().item():.6f}")
    
    # L2 norms
    norms = torch.norm(embed_tensor, dim=1)
    print(f"\nL2 Norms:")
    print(f"  Mean: {norms.mean().item():.6f}")
    print(f"  Std: {norms.std().item():.6f}")
    print(f"  Min: {norms.min().item():.6f}")
    print(f"  Max: {norms.max().item():.6f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution of embedding values
    axes[0, 0].hist(embed_tensor.flatten().numpy(), bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Embedding Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Embedding Values')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # L2 norm distribution
    axes[0, 1].hist(norms.numpy(), bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('L2 Norm')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('L2 Norm Distribution')
    axes[0, 1].axvline(norms.mean().item(), color='red', linestyle='--', label=f'Mean: {norms.mean().item():.2f}')
    axes[0, 1].legend()
    
    # Mean embedding per dimension
    mean_per_dim = embed_tensor.mean(dim=0)
    axes[1, 0].plot(mean_per_dim.numpy(), linewidth=1)
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_title('Mean Embedding Value per Dimension')
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Std per dimension
    std_per_dim = embed_tensor.std(dim=0)
    axes[1, 1].plot(std_per_dim.numpy(), linewidth=1, color='orange')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Std Deviation per Dimension')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, f'{split}_embedding_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved embedding analysis plot to eda_results/{split}_embedding_analysis.png")
    plt.close()
    
    # Compute and visualize similarity matrix (sample)
    print(f"\nComputing pairwise similarities (first 100 samples)...")
    sample_embeds = embed_tensor[:100]
    # Normalize for cosine similarity
    sample_embeds_norm = sample_embeds / sample_embeds.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(sample_embeds_norm, sample_embeds_norm.t())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    ax.set_title('Cosine Similarity Matrix (First 100 Embeddings)')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, f'{split}_similarity_matrix.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved similarity matrix to eda_results/{split}_similarity_matrix.png")
    plt.close()


def visualize_sample_images(data, split='train', num_samples=12):
    """Visualize random sample images with their captions."""
    print(f"\n{'='*70}")
    print(f"SAMPLE IMAGES - {split.upper()} SPLIT")
    print(f"{'='*70}")
    
    images_dir = TRAIN_IMAGES_DIR if split == 'train' else VAL_IMAGES_DIR
    
    # Create caption mapping
    image_to_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_captions:
            image_to_captions[img_id] = []
        image_to_captions[img_id].append(ann['caption'])
    
    # Sample random images
    import random
    random.seed(42)
    sampled_images = random.sample(data['images'], min(num_samples, len(data['images'])))
    
    # Create grid
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, img_info in enumerate(sampled_images):
        if idx >= len(axes):
            break
        
        # Load image
        img_path = os.path.join(images_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Display
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Get first caption
        img_id = img_info['id']
        captions = image_to_captions.get(img_id, ["No caption"])
        caption = captions[0][:80] + "..." if len(captions[0]) > 80 else captions[0]
        
        axes[idx].set_title(f"ID: {img_id}\n{caption}", fontsize=9, wrap=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, f'{split}_sample_images.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved sample images to eda_results/{split}_sample_images.png")
    plt.close()


def compare_train_val():
    """Compare statistics between train and validation splits."""
    print(f"\n{'='*70}")
    print(f"TRAIN vs VALIDATION COMPARISON")
    print(f"{'='*70}")
    
    train_data, train_embeds = load_data('train')
    val_data, val_embeds = load_data('val')
    
    # Caption length comparison
    train_captions = [ann['caption'] for ann in train_data['annotations']]
    val_captions = [ann['caption'] for ann in val_data['annotations']]
    
    train_word_counts = [len(cap.split()) for cap in train_captions]
    val_word_counts = [len(cap.split()) for cap in val_captions]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Caption length comparison
    axes[0].hist(train_word_counts, bins=30, alpha=0.6, label='Train', color='blue', edgecolor='black')
    axes[0].hist(val_word_counts, bins=30, alpha=0.6, label='Val', color='orange', edgecolor='black')
    axes[0].set_xlabel('Caption Length (words)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Caption Length Distribution: Train vs Val')
    axes[0].legend()
    
    # Embedding norm comparison
    train_embed_tensor = torch.stack([train_embeds[k] for k in sorted(train_embeds.keys())])
    val_embed_tensor = torch.stack([val_embeds[k] for k in sorted(val_embeds.keys())])
    
    train_norms = torch.norm(train_embed_tensor, dim=1).numpy()
    val_norms = torch.norm(val_embed_tensor, dim=1).numpy()
    
    axes[1].hist(train_norms, bins=50, alpha=0.6, label='Train', color='blue', edgecolor='black')
    axes[1].hist(val_norms, bins=50, alpha=0.6, label='Val', color='orange', edgecolor='black')
    axes[1].set_xlabel('L2 Norm')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Embedding L2 Norm Distribution: Train vs Val')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'train_vs_val_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to eda_results/train_vs_val_comparison.png")
    plt.close()


def generate_summary_report():
    """Generate a text summary report."""
    print(f"\n{'='*70}")
    print(f"GENERATING SUMMARY REPORT")
    print(f"{'='*70}")
    
    train_data, train_embeds = load_data('train')
    val_data, val_embeds = load_data('val')
    
    report = []
    report.append("="*70)
    report.append("COCO-CLIP DATASET - EXPLORATORY DATA ANALYSIS SUMMARY")
    report.append("="*70)
    report.append("")
    
    # Dataset overview
    report.append("DATASET OVERVIEW")
    report.append("-" * 70)
    report.append(f"Train images: {len(train_data['images']):,}")
    report.append(f"Train captions: {len(train_data['annotations']):,}")
    report.append(f"Val images: {len(val_data['images']):,}")
    report.append(f"Val captions: {len(val_data['annotations']):,}")
    report.append(f"Total images: {len(train_data['images']) + len(val_data['images']):,}")
    report.append(f"Total captions: {len(train_data['annotations']) + len(val_data['annotations']):,}")
    report.append("")
    
    # Caption statistics
    train_captions = [ann['caption'] for ann in train_data['annotations']]
    train_word_counts = [len(cap.split()) for cap in train_captions]
    
    report.append("CAPTION STATISTICS")
    report.append("-" * 70)
    report.append(f"Avg caption length (words): {np.mean(train_word_counts):.1f}")
    report.append(f"Min caption length: {min(train_word_counts)} words")
    report.append(f"Max caption length: {max(train_word_counts)} words")
    report.append("")
    
    # Embedding statistics
    train_embed_tensor = torch.stack([train_embeds[k] for k in sorted(train_embeds.keys())])
    
    report.append("EMBEDDING STATISTICS")
    report.append("-" * 70)
    report.append(f"Embedding dimension: {train_embed_tensor.shape[1]}")
    report.append(f"Total embeddings: {len(train_embeds) + len(val_embeds):,}")
    report.append(f"Total memory: {(train_embed_tensor.element_size() * train_embed_tensor.nelement() + val_embed_tensor.element_size() * val_embed_tensor.nelement()) / (1024**2):.2f} MB")
    report.append("")
    
    # Storage info
    report.append("STORAGE LOCATIONS")
    report.append("-" * 70)
    report.append(f"Images: datasets/coco_subset/{{train,val}}/images/")
    report.append(f"Captions: datasets/coco_subset/{{train,val}}/captions.json")
    report.append(f"Embeddings: datasets/cache/{{train,val}}_text_embeds.pt")
    report.append("")
    
    report.append("="*70)
    report.append("All visualizations saved to: datasets/eda_results/")
    report.append("="*70)
    
    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(EDA_OUTPUT_DIR, 'eda_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Saved summary report to eda_results/eda_summary_report.txt")


def main():
    """Run complete EDA pipeline."""
    print("="*70)
    print("COCO-CLIP DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    print(f"Output directory: {EDA_OUTPUT_DIR}")
    
    # Train split analysis
    train_data, train_embeds = analyze_basic_stats('train')
    analyze_image_properties(train_data, 'train')
    analyze_caption_statistics(train_data, 'train')
    analyze_embeddings(train_embeds, 'train')
    visualize_sample_images(train_data, 'train', num_samples=12)
    
    # Val split analysis
    val_data, val_embeds = analyze_basic_stats('val')
    analyze_image_properties(val_data, 'val')
    analyze_caption_statistics(val_data, 'val')
    analyze_embeddings(val_embeds, 'val')
    visualize_sample_images(val_data, 'val', num_samples=12)
    
    # Comparison
    compare_train_val()
    
    # Summary report
    generate_summary_report()
    
    print(f"\n{'='*70}")
    print("✓ EDA COMPLETE!")
    print(f"{'='*70}")
    print(f"All results saved to: {EDA_OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(os.listdir(EDA_OUTPUT_DIR)):
        print(f"  - {file}")


if __name__ == "__main__":
    main()

