"""
CLIP Training Script with InfoNCE Loss

This script trains a CLIP model using contrastive learning with:
- InfoNCE loss (symmetric contrastive loss over image-text pairs)
- Learnable temperature parameter τ
- Cached text embeddings for efficiency
- AdamW optimizer with weight decay
- Training and validation loops with loss logging
- Loss curve visualization
"""

import os
import sys
import time
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.clip_model import create_clip_model
from datasets.prepare_coco_clip_dataset import COCOCLIPDataset


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive) Loss for CLIP.
    
    This implements the symmetric contrastive loss as described in CLIP:
    - Computes cosine similarity between image and text embeddings
    - Applies temperature scaling: logits = similarity * exp(τ)
    - Uses cross-entropy loss in both directions (image->text and text->image)
    - Averages the two losses for symmetry
    
    The temperature τ is a learnable parameter that controls the sharpness
    of the similarity distribution.
    """
    
    def __init__(self, init_temperature=0.07):
        """
        Initialize InfoNCE loss with learnable temperature.
        
        Args:
            init_temperature (float): Initial value for temperature parameter
        """
        super(InfoNCELoss, self).__init__()
        
        # Learnable temperature parameter (log-space for numerical stability)
        # We use log(τ) and then exp() it to ensure τ > 0
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        
    @property
    def temperature(self):
        """Get the current temperature value."""
        return torch.exp(self.log_temperature)
    
    def forward(self, image_embeddings, text_embeddings):
        """
        Compute InfoNCE loss.
        
        Args:
            image_embeddings (torch.Tensor): L2-normalized image embeddings [batch_size, embedding_dim]
            text_embeddings (torch.Tensor): L2-normalized text embeddings [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device
        
        # Compute cosine similarity matrix (since embeddings are L2-normalized)
        # Shape: [batch_size, batch_size]
        # logits[i, j] = similarity between image i and text j
        logits = torch.matmul(image_embeddings, text_embeddings.t())
        
        # Apply temperature scaling: logits = similarity * exp(τ)
        # This makes the distribution sharper (higher confidence) as τ increases
        logits = logits * self.temperature
        
        # Create labels: diagonal elements are positive pairs
        # i-th image should match i-th text
        labels = torch.arange(batch_size, device=device)
        
        # Symmetric loss: compute loss from both perspectives
        # 1. Image-to-text: for each image, predict which text matches
        loss_i2t = F.cross_entropy(logits, labels)
        
        # 2. Text-to-image: for each text, predict which image matches
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        # Average the two losses for symmetry
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss


def get_gpu_info():
    """Get GPU information if available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        return f"{gpu_name} ({gpu_memory:.1f} GB)"
    return "CPU"


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    """
    Train for one epoch.
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, text_embeddings, image_ids) in enumerate(dataloader):
        try:
            # Move data to device
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            # Normalize text embeddings (cached embeddings are not normalized)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            
            # Forward pass: encode images
            image_embeddings = model.encode_image(images)
            
            # Compute loss
            loss = criterion(image_embeddings, text_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability (includes temperature parameter)
            all_params = list(model.parameters()) + [criterion.log_temperature]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f} | τ: {criterion.temperature.item():.4f}")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠ GPU out of memory at batch {batch_idx}. Skipping batch...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        except Exception as e:
            print(f"\n⚠ Error at batch {batch_idx}: {e}")
            raise e
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, text_embeddings, image_ids in dataloader:
            try:
                # Move data to device
                images = images.to(device)
                text_embeddings = text_embeddings.to(device)
                
                # Normalize text embeddings (cached embeddings are not normalized)
                text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
                
                # Forward pass
                image_embeddings = model.encode_image(images)
                
                # Compute loss
                loss = criterion(image_embeddings, text_embeddings)
                
                total_loss += loss.item()
                num_batches += 1
            
            except Exception as e:
                print(f"\n⚠ Validation error: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss


def plot_loss_curves(train_losses, val_losses, temperatures, save_path):
    """
    Plot and save training/validation loss curves and temperature.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        temperatures (list): List of temperature values per epoch
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature curve
    if temperatures:
        ax2.plot(epochs, temperatures, 'g-', label='Temperature τ', linewidth=2, marker='^')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Temperature', fontsize=12)
        ax2.set_title('Temperature Parameter Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved loss curves to {save_path}")
    plt.close()


def generate_training_report(train_losses, val_losses, temperatures, total_time, gpu_info, 
                              args, issues=None, report_path=None):
    """
    Generate a text report with training summary.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        temperatures (list): Temperature values per epoch
        total_time (float): Total training time in seconds
        gpu_info (str): GPU information
        args: Training arguments
        issues (list): List of observed issues
        report_path (str): Path to save report
    """
    if report_path is None:
        report_path = os.path.join(config.PROJECT_ROOT, 'training_report.txt')
    
    if issues is None:
        issues = []
    
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLIP Training Report\n")
        f.write("="*70 + "\n\n")
        
        # Configuration
        f.write("TRAINING CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Batch size:           {args.batch_size}\n")
        f.write(f"Learning rate:        {args.lr}\n")
        f.write(f"Epochs:               {args.epochs}\n")
        f.write(f"Weight decay:         {args.weight_decay}\n")
        f.write(f"Initial temperature:  {args.init_temperature}\n")
        f.write(f"Device:               {args.device if args.device else 'auto'}\n")
        f.write("\n")
        
        # Hardware and timing
        f.write("HARDWARE & TIMING\n")
        f.write("-"*70 + "\n")
        f.write(f"Hardware:             {gpu_info}\n")
        f.write(f"Total training time:  {hours}h {minutes}m {seconds}s\n")
        f.write(f"Total seconds:        {total_time:.2f}s\n")
        f.write("\n")
        
        # Results
        f.write("TRAINING RESULTS\n")
        f.write("-"*70 + "\n")
        if train_losses:
            f.write(f"Initial train loss:   {train_losses[0]:.4f}\n")
            f.write(f"Final train loss:     {train_losses[-1]:.4f}\n")
            f.write(f"Best train loss:      {min(train_losses):.4f} (epoch {train_losses.index(min(train_losses))+1})\n")
        if val_losses:
            f.write(f"Initial val loss:     {val_losses[0]:.4f}\n")
            f.write(f"Final val loss:       {val_losses[-1]:.4f}\n")
            f.write(f"Best val loss:        {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses))+1})\n")
        if temperatures:
            f.write(f"Initial temperature:  {temperatures[0]:.4f}\n")
            f.write(f"Final temperature:    {temperatures[-1]:.4f}\n")
        f.write(f"Epochs completed:     {len(train_losses)}/{args.epochs}\n")
        f.write("\n")
        
        # Issues
        if issues:
            f.write("OBSERVED ISSUES\n")
            f.write("-"*70 + "\n")
            for issue in issues:
                f.write(f"⚠ {issue}\n")
            f.write("\n")
        
        # Loss per epoch
        f.write("LOSS PER EPOCH\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Temperature':<15}\n")
        f.write("-"*70 + "\n")
        max_len = max(len(train_losses), len(val_losses), len(temperatures))
        for i in range(max_len):
            t_loss = train_losses[i] if i < len(train_losses) else "N/A"
            v_loss = val_losses[i] if i < len(val_losses) else "N/A"
            temp = temperatures[i] if i < len(temperatures) else "N/A"
            if isinstance(t_loss, float) and isinstance(v_loss, float) and isinstance(temp, float):
                f.write(f"{i+1:<8} {t_loss:<15.4f} {v_loss:<15.4f} {temp:<15.4f}\n")
            else:
                f.write(f"{i+1:<8} {str(t_loss):<15} {str(v_loss):<15} {str(temp):<15}\n")
    
    print(f"✓ Saved training report to {report_path}")


def check_for_divergence(losses, threshold=100.0):
    """
    Check if training has diverged (loss exploded).
    
    Args:
        losses (list): List of loss values
        threshold (float): Loss threshold for divergence detection
    
    Returns:
        bool: True if divergence detected
    """
    if len(losses) == 0:
        return False
    
    # Check if latest loss is too high
    if losses[-1] > threshold:
        return True
    
    # Check if loss increased dramatically
    if len(losses) >= 2:
        if losses[-1] > losses[-2] * 10:  # 10x increase
            return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Train CLIP model with InfoNCE loss')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--init_temperature', type=float, default=0.07, help='Initial temperature (default: 0.07)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu, default: auto)')
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader num_workers (default: from config)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    print("="*70)
    print("CLIP Training Script")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Initial temperature: {args.init_temperature}")
    print(f"  Device: {device} ({gpu_info})")
    print("="*70)
    
    # Create directories
    config.create_directories()
    
    # Load cached text embeddings
    print("\nLoading cached text embeddings...")
    try:
        train_text_embeddings = torch.load(config.TRAIN_EMBEDDINGS_PATH, map_location='cpu')
        val_text_embeddings = torch.load(config.VAL_EMBEDDINGS_PATH, map_location='cpu')
        print(f"✓ Loaded train embeddings: {len(train_text_embeddings)} samples")
        print(f"✓ Loaded val embeddings: {len(val_text_embeddings)} samples")
    except FileNotFoundError as e:
        print(f"\n❌ Error: Cached embeddings not found!")
        print(f"   Please run: python datasets/prepare_coco_clip_dataset.py")
        print(f"   Expected paths:")
        print(f"     - {config.TRAIN_EMBEDDINGS_PATH}")
        print(f"     - {config.VAL_EMBEDDINGS_PATH}")
        sys.exit(1)
    
    # Create datasets
    print("\nCreating datasets...")
    try:
        train_dataset = COCOCLIPDataset(
            images_dir=config.TRAIN_IMAGES_DIR,
            captions_path=config.TRAIN_CAPTIONS_PATH,
            text_embeddings=train_text_embeddings
        )
        val_dataset = COCOCLIPDataset(
            images_dir=config.VAL_IMAGES_DIR,
            captions_path=config.VAL_CAPTIONS_PATH,
            text_embeddings=val_text_embeddings
        )
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Val dataset: {len(val_dataset)} samples")
    except Exception as e:
        print(f"\n❌ Error creating datasets: {e}")
        sys.exit(1)
    
    # Create dataloaders
    num_workers = args.num_workers if args.num_workers is not None else config.NUM_WORKERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY if device.type == 'cuda' else False
    )
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    
    # Create model (skip text encoder since we use cached embeddings)
    print("\nCreating CLIP model...")
    print("Note: Text encoder will be skipped to save GPU memory (using cached embeddings)")
    try:
        model = create_clip_model(device=device, use_cached_embeddings=True)
    except Exception as e:
        print(f"\n❌ Error creating model: {e}")
        sys.exit(1)
    
    # Create loss function with learnable temperature
    criterion = InfoNCELoss(init_temperature=args.init_temperature).to(device)
    print(f"\n✓ Created InfoNCE loss with initial temperature: {args.init_temperature}")
    
    # Create optimizer (only optimize trainable parameters + temperature)
    trainable_params = list(model.get_trainable_parameters()) + [criterion.log_temperature]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"✓ Created AdamW optimizer (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    train_losses = []
    val_losses = []
    temperatures = []
    issues = []
    start_time = time.time()
    
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch [{epoch}/{args.epochs}]")
            print("-" * 70)
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # Record temperature
            temp_value = criterion.temperature.item()
            temperatures.append(temp_value)
            
            # Log epoch results
            print(f"\nEpoch [{epoch}/{args.epochs}] Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Temperature τ: {temp_value:.4f}")
            
            # Check for divergence
            if check_for_divergence(train_losses):
                issue_msg = f"Training loss diverged at epoch {epoch} (loss: {train_loss:.4f})"
                issues.append(issue_msg)
                print(f"\n⚠ WARNING: {issue_msg}")
                break
            
            # Check for NaN
            if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss)):
                issue_msg = f"NaN loss detected at epoch {epoch}"
                issues.append(issue_msg)
                print(f"\n⚠ WARNING: {issue_msg}")
                break
            
            # Check for temperature instability (large changes)
            if len(temperatures) >= 2:
                temp_change = abs(temperatures[-1] - temperatures[-2])
                if temp_change > 0.1:  # Large temperature jump
                    issue_msg = f"Large temperature change at epoch {epoch} (Δ={temp_change:.4f})"
                    if issue_msg not in issues:
                        issues.append(issue_msg)
                        print(f"⚠ Note: {issue_msg}")
    
    except KeyboardInterrupt:
        issues.append("Training interrupted by user")
        print("\n\n⚠ Training interrupted by user.")
    except Exception as e:
        issues.append(f"Training error: {str(e)}")
        print(f"\n\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Calculate total training time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # Print summary
        print("\n" + "="*70)
        print("Training Summary")
        print("="*70)
        print(f"Total training time: {hours}h {minutes}m {seconds}s")
        print(f"Hardware: {gpu_info}")
        print(f"Final train loss: {train_losses[-1]:.4f}" if train_losses else "N/A")
        print(f"Final val loss: {val_losses[-1]:.4f}" if val_losses else "N/A")
        if train_losses:
            print(f"Final temperature τ: {criterion.temperature.item():.4f}")
        print("="*70)
        
        # Generate plots and report
        if len(train_losses) > 0 and len(val_losses) > 0:
            # Plot loss curves and temperature
            plot_path = os.path.join(config.PROJECT_ROOT, 'loss_curves.png')
            plot_loss_curves(train_losses, val_losses, temperatures, plot_path)
            
            # Generate training report
            generate_training_report(
                train_losses, val_losses, temperatures, total_time, gpu_info, args, issues
            )
        
        print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

