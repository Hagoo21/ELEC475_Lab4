"""
Global Configuration File for ELEC475 Lab 4

Modify these settings to customize dataset preparation, training, and analysis.
All paths are relative to the project root.
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
DATA_DIR = os.path.join(DATASETS_DIR, 'coco_subset')
CACHE_DIR = os.path.join(DATASETS_DIR, 'cache')
ANNOTATIONS_CACHE_DIR = os.path.join(DATASETS_DIR, 'coco_annotations_cache')
EDA_OUTPUT_DIR = os.path.join(DATASETS_DIR, 'eda_results')

# Training paths
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'train')
CHECKPOINTS_DIR = os.path.join(TRAIN_DIR, 'checkpoints')
LOGS_DIR = os.path.join(TRAIN_DIR, 'logs')

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Analysis paths
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'analysis')

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Number of images to download (CHANGE THESE to download more/less data)
TRAIN_SAMPLES = 16000  # Set to None to use full training set (not recommended)
VAL_SAMPLES = 4000    # Set to None to use full validation set (not recommended)

# Random seed for reproducibility
RANDOM_SEED = 42

# Dataset source
COCO_DATASET_NAME = "coco-2014"
COCO_SPLIT_TRAIN = "train"
COCO_SPLIT_VAL = "val"  # Maps to 'validation' in FiftyOne

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

# CLIP image preprocessing parameters
CLIP_IMAGE_SIZE = 224  # Images resized to 224x224 for CLIP

# CLIP normalization statistics (DO NOT CHANGE - these are CLIP standard)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ============================================================================
# TEXT PREPROCESSING & EMBEDDINGS
# ============================================================================

# CLIP text encoder model
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Text tokenization parameters
CLIP_MAX_LENGTH = 77  # CLIP's context length (DO NOT CHANGE)
TOKENIZER_TRUNCATION = True
TOKENIZER_PADDING = "max_length"

# Text embedding dimension (based on CLIP model)
TEXT_EMBEDDING_DIM = 512  # ViT-B/32 uses 512-dim embeddings

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# DataLoader settings
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 0 for Windows, 4+ for Linux/Mac
PIN_MEMORY = True  # Set to False if no GPU

# Training hyperparameters (placeholder for future training scripts)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Device configuration
DEVICE = "cuda"  # or "cpu"
USE_AMP = True  # Automatic Mixed Precision (faster training on GPU)

# ============================================================================
# EDA CONFIGURATION
# ============================================================================

# Number of sample images to display in EDA
EDA_NUM_SAMPLES = 12

# Number of top words to show in word frequency analysis
EDA_TOP_WORDS = 20

# Embedding similarity matrix sample size
EDA_SIMILARITY_SAMPLE_SIZE = 100

# Figure DPI for saved plots
EDA_FIGURE_DPI = 150

# ============================================================================
# LOGGING & MONITORING
# ============================================================================

# Logging level
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Save frequency (for training checkpoints)
SAVE_EVERY_N_EPOCHS = 1
LOG_EVERY_N_STEPS = 100

# Experiment tracking (placeholder)
USE_WANDB = False  # Set to True to enable Weights & Biases logging
WANDB_PROJECT = "elec475-lab4"
WANDB_ENTITY = None  # Your W&B username/team

# ============================================================================
# DERIVED PATHS (Auto-generated - DO NOT MODIFY)
# ============================================================================

# Train split paths
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'images')
TRAIN_CAPTIONS_PATH = os.path.join(DATA_DIR, 'train', 'captions.json')
TRAIN_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, 'train_text_embeds.pt')

# Val split paths
VAL_IMAGES_DIR = os.path.join(DATA_DIR, 'val', 'images')
VAL_CAPTIONS_PATH = os.path.join(DATA_DIR, 'val', 'captions.json')
VAL_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, 'val_text_embeds.pt')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_directories():
    """Create all necessary directories if they don't exist."""
    dirs = [
        DATASETS_DIR,
        DATA_DIR,
        CACHE_DIR,
        ANNOTATIONS_CACHE_DIR,
        EDA_OUTPUT_DIR,
        TRAIN_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        MODELS_DIR,
        ANALYSIS_DIR,
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def print_config():
    """Print current configuration for verification."""
    print("="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    print(f"\nDATASET:")
    print(f"  Train samples: {TRAIN_SAMPLES}")
    print(f"  Val samples: {VAL_SAMPLES}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"\nPATHS:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Cache directory: {CACHE_DIR}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    print(f"\nMODEL:")
    print(f"  CLIP model: {CLIP_MODEL_NAME}")
    print(f"  Image size: {CLIP_IMAGE_SIZE}x{CLIP_IMAGE_SIZE}")
    print(f"  Text embedding dim: {TEXT_EMBEDDING_DIM}")
    print(f"\nTRAINING:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Num epochs: {NUM_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print("="*70)


if __name__ == "__main__":
    # When run directly, print configuration and create directories
    create_directories()
    print_config()
    print("\n✓ All directories created successfully!")

