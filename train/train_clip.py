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
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.clip_model import create_clip_model
from datasets.dataloader import COCOCLIPDataset


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
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "Apple Silicon GPU (MPS)"
    return "CPU"


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs, grad_clip=5.0, log_gradients=False):
    """
    Train for one epoch.
    
    Args:
        log_gradients (bool): If True, log gradient statistics
    
    Returns:
        tuple: (average_loss, gradient_stats_dict)
            - average_loss: Average training loss for the epoch
            - gradient_stats_dict: Dictionary with gradient statistics (if log_gradients=True)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Gradient monitoring
    gradient_norms = []
    clipped_gradient_norms = []
    
    # Create progress bar for training batches
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]", 
                leave=False, ncols=100)
    
    for batch_idx, (images, text_embeddings, image_ids) in enumerate(pbar):
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
            
            # Compute gradient norms before clipping (for monitoring)
            all_params = list(model.parameters()) + [criterion.log_temperature]
            total_grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=float('inf'))
            
            # Gradient clipping for stability (includes temperature parameter)
            # Default increased from 1.0 to 5.0 to allow more learning
            clipped_grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
            
            # Store gradient norms for logging
            if log_gradients:
                gradient_norms.append(total_grad_norm.item())
                clipped_gradient_norms.append(clipped_grad_norm.item())
            
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar with current loss, temperature, and gradient info
            current_loss = loss.item()
            current_temp = criterion.temperature.item()
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'τ': f'{current_temp:.4f}',
                'Grad': f'{total_grad_norm.item():.3f}',
                'Avg': f'{total_loss/num_batches:.4f}'
            })
        
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                pbar.write(f"⚠ GPU out of memory at batch {batch_idx}. Skipping batch...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        except Exception as e:
            # Check if it's a CUDA memory error in the exception message
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                pbar.write(f"⚠ GPU out of memory at batch {batch_idx}. Skipping batch...")
                torch.cuda.empty_cache()
                continue
            pbar.write(f"⚠ Error at batch {batch_idx}: {e}")
            raise e
    
    if num_batches == 0:
        raise RuntimeError(
            "All training batches failed due to GPU out of memory!\n"
            "Please reduce batch size (try --batch_size 8 or --batch_size 16)"
        )
    
    avg_loss = total_loss / num_batches
    
    # Prepare gradient statistics
    grad_stats = {}
    if log_gradients and gradient_norms:
        grad_stats = {
            'mean_grad_norm': sum(gradient_norms) / len(gradient_norms),
            'max_grad_norm': max(gradient_norms),
            'mean_clipped_grad_norm': sum(clipped_gradient_norms) / len(clipped_gradient_norms),
            'clipping_ratio': sum(1 for g, c in zip(gradient_norms, clipped_gradient_norms) if g > c) / len(gradient_norms)
        }
    
    return avg_loss, grad_stats


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Create progress bar for validation batches
    pbar = tqdm(dataloader, desc="Validation", leave=False, ncols=100)
    
    with torch.no_grad():
        for images, text_embeddings, image_ids in pbar:
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
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{total_loss/num_batches:.4f}'
                })
            
            except Exception as e:
                pbar.write(f"⚠ Validation error: {e}")
                continue
    
    if num_batches == 0:
        print("⚠ Warning: All validation batches failed. Returning high loss value.")
        return 999.0
    
    avg_loss = total_loss / num_batches
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
                              args, issues=None, report_path=None, gradient_stats=None):
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
        report_path = os.path.join(config.TRAIN_DIR, 'training_report.txt')
    
    if issues is None:
        issues = []
    
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    with open(report_path, 'w', encoding='utf-8') as f:
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
        f.write(f"Gradient clip norm:   {args.grad_clip}\n")
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
        
        # Gradient statistics
        if gradient_stats and len(gradient_stats) > 0:
            f.write("GRADIENT STATISTICS\n")
            f.write("-"*70 + "\n")
            avg_grad_norm = sum(s.get('mean_grad_norm', 0) for s in gradient_stats) / len(gradient_stats)
            max_grad_norm = max(s.get('max_grad_norm', 0) for s in gradient_stats)
            avg_clipping_ratio = sum(s.get('clipping_ratio', 0) for s in gradient_stats) / len(gradient_stats)
            f.write(f"Average gradient norm: {avg_grad_norm:.3f}\n")
            f.write(f"Max gradient norm:     {max_grad_norm:.3f}\n")
            f.write(f"Gradients clipped:     {100*avg_clipping_ratio:.1f}% of batches\n")
            f.write("\n")
        
        # Issues
        if issues:
            f.write("OBSERVED ISSUES\n")
            f.write("-"*70 + "\n")
            for issue in issues:
                f.write(f"[!] {issue}\n")
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


def save_checkpoint(model, optimizer, criterion, epoch, train_losses, val_losses, temperatures, 
                   best_val_loss, checkpoint_dir, is_best=False, save_optimizer=False):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        criterion: The loss function (for temperature parameter)
        epoch: Current epoch number
        train_losses: List of training losses
        val_losses: List of validation losses
        temperatures: List of temperature values
        best_val_loss: Best validation loss so far
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
        save_optimizer: Whether to save optimizer state (default False - saves space and avoids write errors)
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except Exception as e:
        print(f"⚠ Warning: Failed to create checkpoint directory: {e}")
        return
    
    # Create checkpoint dict (without optimizer state by default to avoid large file issues)
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'temperatures': temperatures,
            'best_val_loss': best_val_loss,
        }
        
        # Only include optimizer state if explicitly requested (it triples the file size)
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    except Exception as e:
        print(f"⚠ Warning: Failed to create checkpoint dict: {e}")
        return
    
    # Save regular checkpoint with atomic write pattern
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    temp_checkpoint_path = checkpoint_path + '.tmp'
    
    try:
        # Save to temporary file first (atomic write pattern prevents corruption)
        torch.save(checkpoint, temp_checkpoint_path)
        # Only rename to final path if save succeeded
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        os.rename(temp_checkpoint_path, checkpoint_path)
        
        # Get file size for reporting
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        opt_note = " (with optimizer)" if save_optimizer else ""
        print(f"✓ Saved checkpoint{opt_note}: {checkpoint_path} ({file_size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_checkpoint_path):
            try:
                os.remove(temp_checkpoint_path)
            except:
                pass
        return
    
    # Save best model separately (also without optimizer state by default)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        temp_best_path = best_path + '.tmp'
        try:
            torch.save(checkpoint, temp_best_path)
            if os.path.exists(best_path):
                os.remove(best_path)
            os.rename(temp_best_path, best_path)
            print(f"✓ Saved best model checkpoint (val_loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"⚠ Warning: Failed to save best checkpoint: {e}")
            # Clean up temp file
            if os.path.exists(temp_best_path):
                try:
                    os.remove(temp_best_path)
                except:
                    pass


def load_checkpoint(checkpoint_path, model, optimizer, criterion, device):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (can be None)
        criterion: Loss function to load state into
        device: Device to load checkpoint on
    
    Returns:
        dict: Checkpoint data (epoch, losses, etc.)
    """
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if available (lightweight checkpoints may not have it)
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ Loaded optimizer state")
        except Exception as e:
            print(f"⚠ Warning: Could not load optimizer state: {e}")
            print(f"  Training will continue with fresh optimizer state")
    else:
        print(f"⚠ Checkpoint does not contain optimizer state (lightweight checkpoint)")
        print(f"  Training will continue with fresh optimizer state")
    
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Format best_val_loss with proper handling for missing values
    best_val = checkpoint.get('best_val_loss', None)
    if best_val is not None:
        print(f"  Best validation loss: {best_val:.4f}")
    else:
        print(f"  Best validation loss: N/A")
    
    return checkpoint


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
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (default: 5e-4, increased from 1e-4 for better learning)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--init_temperature', type=float, default=0.07, help='Initial temperature (default: 0.07)')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping max norm (default: 5.0, increased from 1.0)')
    parser.add_argument('--log_gradients', action='store_true', help='Log gradient statistics to training report')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/mps/cpu, default: auto - prefers MPS on Mac, CUDA on Linux/Windows)')
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader num_workers (default: from config)')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit dataset to N samples for quick testing (default: None, use full dataset)')
    parser.add_argument('--train_percentage', type=float, default=None, help='Use X%% of train dataset (e.g., 20 for 20%%, default: None, use full dataset)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from (default: None)')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs (default: 1)')
    parser.add_argument('--save_optimizer', action='store_true', help='Save optimizer state in checkpoints (increases file size ~3x)')
    parser.add_argument('--no_pin_memory', action='store_true', help='Disable pin_memory (useful for smaller GPUs)')
    
    args = parser.parse_args()
    
    # Set device with automatic selection (MPS > CUDA > CPU)
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # Clear GPU cache if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"✓ Cleared CUDA GPU cache")
    elif device.type == 'mps':
        # MPS doesn't need explicit cache clearing, but we can note it's being used
        pass
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    # Auto-disable pin_memory for smaller CUDA GPUs (< 8GB) unless explicitly enabled
    # Note: MPS doesn't support pin_memory, so it's automatically disabled for MPS
    if device.type == 'cuda' and not args.no_pin_memory:
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 8.0:
                print(f"⚠ GPU has {gpu_memory_gb:.1f}GB - auto-disabling pin_memory for better memory usage")
                args.no_pin_memory = True
        except:
            pass
    
    # MPS doesn't support pin_memory, so disable it automatically
    use_pin_memory = config.PIN_MEMORY and device.type == 'cuda' and not args.no_pin_memory
    if device.type == 'mps':
        use_pin_memory = False
    
    print("="*70)
    print("CLIP Training Script")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Gradient clip norm: {args.grad_clip}")
    print(f"  Initial temperature: {args.init_temperature}")
    if args.log_gradients:
        print(f"  Gradient logging: ENABLED")
    if args.train_percentage is not None:
        print(f"  Train percentage: {args.train_percentage}%")
    if args.max_samples is not None:
        print(f"  Max samples (testing): {args.max_samples}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print(f"  Save every: {args.save_every} epoch(s)")
    print(f"  Pin memory: {use_pin_memory}")
    print(f"  Device: {device} ({gpu_info})")
    if device.type == 'cuda' and args.batch_size >= 32:
        print(f"\n⚠ WARNING: Batch size {args.batch_size} may be too large for 6GB GPU!")
        print(f"   Consider using --batch_size 8 or --batch_size 16")
    elif device.type == 'mps':
        if args.batch_size > 32:
            print(f"\n⚠ Note: Using MPS (Apple Silicon GPU). Large batch sizes may cause memory issues.")
            print(f"   If you encounter OOM errors, try reducing batch size.")
    print("="*70)
    
    # Create directories
    config.create_directories()
    
    # Ensure cached text embeddings exist
    print("\nChecking cached text embeddings...")
    missing_paths = [
        path for path in (config.TRAIN_EMBEDDINGS_PATH, config.VAL_EMBEDDINGS_PATH)
        if not os.path.exists(path)
    ]
    if missing_paths:
        print(f"\n❌ Error: Cached embeddings not found!")
        print(f"   Please run: python datasets/preprocess_data.py")
        print(f"   Missing:")
        for path in missing_paths:
            print(f"     - {path}")
        sys.exit(1)
    print("✓ Found cached embeddings for train/val splits")
    
    # Create datasets
    print("\nCreating datasets...")
    try:
        train_dataset = COCOCLIPDataset(
            images_dir=config.TRAIN_IMAGES_DIR,
            captions_path=config.TRAIN_CAPTIONS_PATH,
            text_embeddings=config.TRAIN_EMBEDDINGS_PATH
        )
        val_dataset = COCOCLIPDataset(
            images_dir=config.VAL_IMAGES_DIR,
            captions_path=config.VAL_CAPTIONS_PATH,
            text_embeddings=config.VAL_EMBEDDINGS_PATH
        )
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Val dataset: {len(val_dataset)} samples")
    except Exception as e:
        print(f"\n❌ Error creating datasets: {e}")
        sys.exit(1)
    
    # Limit datasets for quick testing if requested
    # Limit datasets if requested (either by max_samples or percentage)
    if args.train_percentage is not None:
        if args.train_percentage <= 0 or args.train_percentage > 100:
            print(f"\n❌ Error: --train_percentage must be between 0 and 100 (got {args.train_percentage})")
            sys.exit(1)
        train_samples = int(len(train_dataset) * args.train_percentage / 100)
        val_samples = int(len(val_dataset) * args.train_percentage / 100)
        print(f"\n⚠ Using {args.train_percentage}% of datasets...")
        print(f"  Train: {train_samples} samples (out of {len(train_dataset)})")
        print(f"  Val: {val_samples} samples (out of {len(val_dataset)})")
        train_indices = list(range(train_samples))
        val_indices = list(range(val_samples))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"✓ Train dataset limited to: {len(train_dataset)} samples")
        print(f"✓ Val dataset limited to: {len(val_dataset)} samples")
    elif args.max_samples is not None:
        print(f"\n⚠ Limiting datasets to {args.max_samples} samples for quick testing...")
        train_indices = list(range(min(args.max_samples, len(train_dataset))))
        val_indices = list(range(min(args.max_samples, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"✓ Train dataset limited to: {len(train_dataset)} samples")
        print(f"✓ Val dataset limited to: {len(val_dataset)} samples")
    
    # Create dataloaders
    num_workers = args.num_workers if args.num_workers is not None else config.NUM_WORKERS
    
    # Try creating dataloaders with pin_memory, fallback if it fails
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
        # Test if we can actually load a batch (catches pin_memory OOM early)
        if use_pin_memory and device.type == 'cuda':
            try:
                _ = next(iter(train_loader))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"\n⚠ Pin memory causes OOM, disabling it...")
                    use_pin_memory = False
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=False
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=False
                    )
    except Exception as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n⚠ OOM during DataLoader creation, disabling pin_memory and retrying...")
            use_pin_memory = False
            torch.cuda.empty_cache()
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False
            )
        else:
            raise e
    
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
    
    # Initialize training state
    start_epoch = 1
    train_losses = []
    val_losses = []
    temperatures = []
    gradient_stats = []  # Store gradient statistics per epoch
    best_val_loss = float('inf')
    issues = []
    
    # Resume from checkpoint if specified
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"\n❌ Error: Checkpoint file not found: {args.resume}")
            sys.exit(1)
        checkpoint_data = load_checkpoint(args.resume, model, optimizer, criterion, device)
        start_epoch = checkpoint_data['epoch'] + 1
        train_losses = checkpoint_data.get('train_losses', [])
        val_losses = checkpoint_data.get('val_losses', [])
        temperatures = checkpoint_data.get('temperatures', [])
        best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
        print(f"✓ Resuming training from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Create overall progress bar for epochs
        epoch_pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Training Progress", ncols=100)
        
        for epoch in epoch_pbar:
            # Train
            train_loss, grad_stats = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.epochs,
                grad_clip=args.grad_clip, log_gradients=args.log_gradients
            )
            train_losses.append(train_loss)
            if grad_stats:
                gradient_stats.append(grad_stats)
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # Record temperature
            temp_value = criterion.temperature.item()
            temperatures.append(temp_value)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'τ': f'{temp_value:.4f}'
            })
            
            # Log epoch results
            epoch_pbar.write(f"\nEpoch [{epoch}/{args.epochs}] Summary:")
            epoch_pbar.write(f"  Train Loss: {train_loss:.4f}")
            epoch_pbar.write(f"  Val Loss:   {val_loss:.4f}")
            epoch_pbar.write(f"  Temperature τ: {temp_value:.4f}")
            if grad_stats:
                epoch_pbar.write(f"  Gradient Norm: {grad_stats['mean_grad_norm']:.3f} (clipped: {grad_stats['mean_clipped_grad_norm']:.3f})")
                if grad_stats['clipping_ratio'] > 0:
                    epoch_pbar.write(f"  Gradients clipped: {100*grad_stats['clipping_ratio']:.1f}% of batches")
            
            # Check if this is the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epoch_pbar.write(f"  🎯 New best validation loss!")
            
            # Save checkpoint
            if epoch % args.save_every == 0 or is_best:
                save_checkpoint(
                    model, optimizer, criterion, epoch, train_losses, val_losses, temperatures,
                    best_val_loss, config.CHECKPOINTS_DIR, is_best=is_best, 
                    save_optimizer=args.save_optimizer
                )
            
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
            plot_path = os.path.join(config.TRAIN_DIR, 'loss_curves.png')
            plot_loss_curves(train_losses, val_losses, temperatures, plot_path)
            
            # Generate training report
            generate_training_report(
                train_losses, val_losses, temperatures, total_time, gpu_info, args, issues,
                gradient_stats=gradient_stats if args.log_gradients else None
            )
        
        print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
