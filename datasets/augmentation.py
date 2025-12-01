"""
Data Augmentation Module for CLIP Training
ELEC475 Lab 4 - Section 2.5

This module provides data augmentation transforms to improve CLIP model generalization.
Augmentation is Modification 5 in the ablation study.

AUGMENTATION STRATEGIES:
1. Baseline: Minimal augmentation (resize + center crop + normalize)
2. Basic: Add random horizontal flip
3. Advanced: Add random resized crop + color jitter
4. Strong: Aggressive augmentation with AutoAugment-style transforms
"""

import torch
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_augmentation_transforms(augmentation_level="baseline", image_size=224):
    """
    Get image augmentation transforms based on augmentation level.
    
    Args:
        augmentation_level (str): Level of augmentation
            - "baseline": Minimal (resize + center crop + normalize)
            - "basic": Add random horizontal flip
            - "advanced": Add random resized crop + color jitter
            - "strong": Add stronger color augmentation and random rotation
        image_size (int): Target image size (default: 224)
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    
    # CLIP normalization (DO NOT CHANGE)
    normalize = transforms.Normalize(
        mean=config.CLIP_MEAN,
        std=config.CLIP_STD
    )
    
    if augmentation_level == "baseline":
        # Baseline: No augmentation, just resize and normalize
        # Used in original baseline model
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_level == "basic":
        # Basic augmentation: Add random horizontal flip
        # Minimal data augmentation for initial testing
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_level == "advanced":
        # Advanced augmentation: Random resized crop + color jitter + flip
        # This is a standard augmentation strategy for image encoders
        return transforms.Compose([
            # Random resized crop: crop 0.8-1.0 of image, then resize to target
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            # Color jitter: subtle changes to brightness, contrast, saturation
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_level == "strong":
        # Strong augmentation: Aggressive transforms for robust learning
        # May hurt performance if too aggressive
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            # Stronger color jitter
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            # Random rotation (small angles)
            transforms.RandomRotation(degrees=10),
            # Random grayscale conversion (10% chance)
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize
        ])
    
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}. "
                        f"Choose from: baseline, basic, advanced, strong")


def get_validation_transforms(image_size=224):
    """
    Get transforms for validation/test (no augmentation).
    
    Args:
        image_size (int): Target image size
    
    Returns:
        torchvision.transforms.Compose: Validation transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.CLIP_MEAN,
            std=config.CLIP_STD
        )
    ])


def print_augmentation_info(augmentation_level):
    """
    Print information about the augmentation strategy.
    
    Args:
        augmentation_level (str): Augmentation level name
    """
    info = {
        "baseline": {
            "name": "Baseline (No Augmentation)",
            "transforms": [
                "Resize to 224x224",
                "Convert to tensor",
                "Normalize with CLIP statistics"
            ],
            "description": "Minimal preprocessing, no augmentation. Used in baseline model."
        },
        "basic": {
            "name": "Basic Augmentation",
            "transforms": [
                "Resize to 224x224",
                "Random horizontal flip (p=0.5)",
                "Convert to tensor",
                "Normalize with CLIP statistics"
            ],
            "description": "Light augmentation with only horizontal flips."
        },
        "advanced": {
            "name": "Advanced Augmentation",
            "transforms": [
                "Random resized crop (scale=0.8-1.0, ratio=0.9-1.1)",
                "Random horizontal flip (p=0.5)",
                "Color jitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)",
                "Convert to tensor",
                "Normalize with CLIP statistics"
            ],
            "description": "Standard augmentation strategy for robust vision models."
        },
        "strong": {
            "name": "Strong Augmentation",
            "transforms": [
                "Random resized crop (scale=0.7-1.0, ratio=0.8-1.2)",
                "Random horizontal flip (p=0.5)",
                "Strong color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)",
                "Random rotation (±10 degrees)",
                "Random grayscale (p=0.1)",
                "Convert to tensor",
                "Normalize with CLIP statistics"
            ],
            "description": "Aggressive augmentation for maximum regularization."
        }
    }
    
    if augmentation_level not in info:
        print(f"Unknown augmentation level: {augmentation_level}")
        return
    
    aug_info = info[augmentation_level]
    print(f"\n{'='*70}")
    print(f"DATA AUGMENTATION: {aug_info['name']}")
    print(f"{'='*70}")
    print(f"Description: {aug_info['description']}")
    print(f"\nTransforms applied:")
    for i, transform in enumerate(aug_info['transforms'], 1):
        print(f"  {i}. {transform}")
    print(f"{'='*70}\n")


# ============================================================================
# AUGMENTATION CONFIGURATIONS FOR ABLATION STUDY
# ============================================================================

def get_augmentation_configs():
    """
    Get predefined augmentation configurations for ablation study.
    
    Returns:
        dict: Configuration name -> augmentation level
    """
    configs = {
        # Baseline experiments (no augmentation or minimal augmentation)
        "no_aug": {
            "train_aug": "baseline",
            "val_aug": "baseline",
            "description": "No augmentation (baseline)"
        },
        
        # Basic augmentation experiments
        "basic_aug": {
            "train_aug": "basic",
            "val_aug": "baseline",
            "description": "Basic augmentation (horizontal flip only)"
        },
        
        # Advanced augmentation experiments (recommended)
        "advanced_aug": {
            "train_aug": "advanced",
            "val_aug": "baseline",
            "description": "Advanced augmentation (crop + color jitter + flip)"
        },
        
        # Strong augmentation experiments
        "strong_aug": {
            "train_aug": "strong",
            "val_aug": "baseline",
            "description": "Strong augmentation (aggressive transforms)"
        }
    }
    
    return configs


if __name__ == "__main__":
    """
    Test script to visualize augmentation effects.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    print("Testing Data Augmentation Module...\n")
    
    # Print available configurations
    configs = get_augmentation_configs()
    print(f"Available augmentation configurations: {len(configs)}")
    print("="*70)
    for name, cfg in configs.items():
        print(f"{name:20s} - {cfg['description']}")
    print("="*70 + "\n")
    
    # Print info for each augmentation level
    for level in ["baseline", "basic", "advanced", "strong"]:
        print_augmentation_info(level)
    
    # Test transforms (create dummy image)
    print("\nTesting transform initialization...")
    for level in ["baseline", "basic", "advanced", "strong"]:
        try:
            transform = get_augmentation_transforms(level)
            print(f"✓ {level:15s} transform created successfully")
        except Exception as e:
            print(f"✗ {level:15s} failed: {e}")
    
    # Test validation transform
    try:
        val_transform = get_validation_transforms()
        print(f"✓ {'validation':15s} transform created successfully")
    except Exception as e:
        print(f"✗ {'validation':15s} failed: {e}")
    
    print("\n[OK] Data augmentation module test complete!")
    print("\nNote: To visualize augmentation effects on real images, use:")
    print("  python datasets/visualize_augmentation.py")
