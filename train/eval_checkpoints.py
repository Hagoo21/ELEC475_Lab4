"""
Evaluation script for ELEC475 Lab 4

Purpose: Load baseline (b_*), modified v1 (m_*), and modified v2 (m2_*) checkpoints
from `train/checkpoints/`, restore model state, and compute/report:
- Validation loss
- Recall@{1,5,10} for image→text and text→image
- Optional: a few qualitative retrieval examples

Outputs:
- Summary TXT per model family under `train/eval_results/`
- CSV with Recall@K per model under `train/eval_results/`
- Minimal qualitative TXT per model under `train/eval_results/`

Usage:
  python train/eval_checkpoints.py --device auto

Notes:
- Uses cached text embeddings; text encoder is skipped
- Assumes checkpoint filename prefixes: b_*, m_*, m2_*
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from datasets.dataloader import COCOCLIPDataset
from train.train_clip import compute_recall_at_k, validate, InfoNCELoss
from models.clip_model import create_clip_model
from models.clip_model_modified import create_modified_clip_model
from models.clip_model_modified_2 import create_modified2_clip_model
from models.clip_model_modified_3 import create_modified3_clip_model


def list_checkpoints(checkpoints_dir: str):
    files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')])
    groups = {
        'baseline': [f for f in files if f.startswith('b_')],
        'modified': [f for f in files if f.startswith('m_')],
        'modified2': [f for f in files if f.startswith('m2_')],
        'modified3': [f for f in files if f.startswith('checkpoint')],
    }
    return groups


def load_checkpoint_state(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def build_dataloaders(batch_size: int, num_workers: int, use_pin_memory: bool, subset_percentage: float = None):
    val_dataset = COCOCLIPDataset(
        images_dir=config.VAL_IMAGES_DIR,
        captions_path=config.VAL_CAPTIONS_PATH,
        text_embeddings=config.VAL_EMBEDDINGS_PATH,
        transform=None,
    )
    if subset_percentage is not None:
        subset_n = int(len(val_dataset) * subset_percentage / 100.0)
        val_dataset = Subset(val_dataset, list(range(subset_n)))

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    return val_loader


def eval_one_checkpoint(model, criterion, val_loader, device, save_dir: str, tag: str):
    os.makedirs(save_dir, exist_ok=True)

    # Validation loss
    val_loss = validate(model, val_loader, criterion, device)

    # Recall@K
    recall = compute_recall_at_k(model, val_loader, device, ks=(1, 5, 10))

    # Save summary TXT
    txt_path = os.path.join(save_dir, f'{tag}_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Evaluation Summary\n')
        f.write(f'Checkpoint tag: {tag}\n')
        f.write(f'Val loss: {val_loss:.4f}\n')
        if recall:
            f.write(f"Num samples: {recall['num_samples']}\n")
            for k, v in recall['i2t'].items():
                f.write(f"i→t Recall@{k}: {v*100:.2f}%\n")
            for k, v in recall['t2i'].items():
                f.write(f"t→i Recall@{k}: {v*100:.2f}%\n")

    # Save CSV
    if recall:
        csv_path = os.path.join(save_dir, f'{tag}_recall.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('Metric,K,Value,NumSamples\n')
            for k, v in recall['i2t'].items():
                f.write(f'Image->Text,{k},{v:.6f},{recall["num_samples"]}\n')
            for k, v in recall['t2i'].items():
                f.write(f'Text->Image,{k},{v:.6f},{recall["num_samples"]}\n')

    # Qualitative: sample 3 val images and list top-5 captions (IDs only)
    # We reuse embeddings computed in recall function; for simplicity, run a small pass here
    # (kept brief; detailed visualization not required by lab)
    try:
        # Pull a small batch
        val_iter = iter(val_loader)
        images, text_embeds, image_ids = next(val_iter)
        images = images.to(device)
        with torch.no_grad():
            img_emb = F.normalize(model.encode_image(images), p=2, dim=1)
            txt_emb = F.normalize(text_embeds.to(device), p=2, dim=1)
            sims = img_emb @ txt_emb.t()  # [B, N]
            topk = sims.topk(5, dim=1).indices.cpu().tolist()
        qual_path = os.path.join(save_dir, f'{tag}_qualitative.txt')
        with open(qual_path, 'w', encoding='utf-8') as f:
            f.write('Qualitative Retrieval (first batch, top-5 captions per image)\n')
            for i, img_id in enumerate(image_ids[:min(3, len(image_ids))]):
                f.write(f'ImageID={img_id} -> top5 caption indices: {topk[i][:5]}\n')
    except Exception:
        pass

    return val_loss, recall


def create_model_for_family(family: str, device: torch.device):
    if family == 'baseline':
        model = create_clip_model(device=device, use_cached_embeddings=True)
    elif family == 'modified':
        model = create_modified_clip_model(config_name='layer_norm', device=device, use_cached_embeddings=True)
    elif family == 'modified2':
        model = create_modified2_clip_model(config_name='best_combo_v2', device=device, use_cached_embeddings=True)
    elif family == 'modified3':
        model = create_modified3_clip_model(config_name='best_v3_default', device=device, use_cached_embeddings=True)
    else:
        raise ValueError(f'Unknown family: {family}')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto', help='cuda/mps/cpu or auto')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--subset_percentage', type=float, default=None, help='Evaluate on subset of val (e.g., 20)')
    args = parser.parse_args()

    # Device
    if args.device == 'auto' or args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Dirs
    ckpt_dir = config.CHECKPOINTS_DIR
    out_dir = os.path.join(config.TRAIN_DIR, 'eval_results')
    os.makedirs(out_dir, exist_ok=True)

    # Data
    num_workers = args.num_workers if args.num_workers is not None else config.NUM_WORKERS
    val_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=num_workers,
        use_pin_memory=args.pin_memory,
        subset_percentage=args.subset_percentage,
    )

    # Group checkpoints
    groups = list_checkpoints(ckpt_dir)

    # Evaluate latest checkpoint per family (best for report brevity)
    for family, flist in groups.items():
        if not flist:
            continue
        # Pick best if exists; else last epoch by sort
        # Prefer files named with 'best' if present
        best_candidates = [f for f in flist if 'best' in f]
        ckpt_name = best_candidates[0] if best_candidates else flist[-1]
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        tag = f'{family}_{Path(ckpt_name).stem}'

        print(f"\nEvaluating {family}: {ckpt_name}")

        # Build model
        model = create_model_for_family(family, device)
        # Criterion with temperature (state loaded from checkpoint)
        criterion = InfoNCELoss(init_temperature=0.07).to(device)

        # Load state
        ckpt = load_checkpoint_state(ckpt_path, device)
        model.load_state_dict(ckpt['model_state_dict'])
        try:
            criterion.load_state_dict(ckpt['criterion_state_dict'])
        except Exception:
            pass

        # Eval
        save_dir = os.path.join(out_dir, family)
        eval_one_checkpoint(model, criterion, val_loader, device, save_dir, tag)

    print("\n✓ Evaluation complete. See train/eval_results/")


if __name__ == '__main__':
    main()
