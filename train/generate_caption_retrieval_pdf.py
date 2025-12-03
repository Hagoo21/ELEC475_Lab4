"""
Generate a PDF per model showing the query caption and top-5 retrieved images.

Models supported (families): baseline (b_*), modified (m_*), modified2 (m2_*), modified3 (checkpoint*).

Usage:
  python3 train/generate_caption_retrieval_pdf.py --caption "a cat sitting on a couch" --families baseline modified modified2 modified3 --device auto

Outputs:
  train/eval_results/<family>/<family>_top5_<slug>.pdf

Notes:
- Uses cached text embeddings for data loading; computes image embeddings with the selected model.
- Computes the query caption embedding using HuggingFace CLIP text encoder (CPU fallback if needed).
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from datasets.dataloader import COCOCLIPDataset
from models.clip_model import create_clip_model
from models.clip_model_modified import create_modified_clip_model
from models.clip_model_modified_2 import create_modified2_clip_model
from models.clip_model_modified_3 import create_modified3_clip_model

# Text encoder (for query caption)
from transformers import CLIPTokenizer, CLIPModel


def slugify(text: str, max_len: int = 40) -> str:
    s = ''.join(c if c.isalnum() else '_' for c in text.strip())
    s = '_'.join([p for p in s.split('_') if p])
    return s[:max_len]


def list_checkpoints(checkpoints_dir: str) -> Dict[str, List[str]]:
    files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')])
    return {
        'baseline': [f for f in files if f.startswith('b_')],
        'modified': [f for f in files if f.startswith('m_')],
        'modified2': [f for f in files if f.startswith('m2_')],
        'modified3': [f for f in files if f.startswith('checkpoint')],
    }


def pick_best_or_latest(flist: List[str]) -> str:
    if not flist:
        return ''
    best = [f for f in flist if 'best' in f]
    return best[0] if best else flist[-1]


def build_val_loader(batch_size: int, num_workers: int, pin_memory: bool, subset_percentage: float = None) -> Tuple[DataLoader, Dict[int, str]]:
    val_dataset = COCOCLIPDataset(
        images_dir=config.VAL_IMAGES_DIR,
        captions_path=config.VAL_CAPTIONS_PATH,
        text_embeddings=config.VAL_EMBEDDINGS_PATH,
        transform=None,
    )
    if subset_percentage is not None and subset_percentage > 0 and subset_percentage <= 100:
        n = int(len(val_dataset) * subset_percentage / 100.0)
        val_dataset = Subset(val_dataset, list(range(n)))
        # When using Subset, id->path mapping is trickier; rebuild mapping via indices
        # Instead, build mapping from the underlying dataset
        base = val_dataset.dataset  # type: ignore
        id_to_path = {img['id']: os.path.join(base.images_dir, img['file_name']) for img in base.images}
    else:
        id_to_path = {img['id']: os.path.join(val_dataset.images_dir, img['file_name']) for img in val_dataset.images}  # type: ignore

    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, id_to_path


def create_model_for_family(family: str, device: torch.device):
    if family == 'baseline':
        return create_clip_model(device=device, use_cached_embeddings=True)
    if family == 'modified':
        return create_modified_clip_model(config_name='layer_norm', device=device, use_cached_embeddings=True)
    if family == 'modified2':
        return create_modified2_clip_model(config_name='best_combo_v2', device=device, use_cached_embeddings=True)
    if family == 'modified3':
        return create_modified3_clip_model(config_name='convnext_tiny_v3', device=device, use_cached_embeddings=True)
    raise ValueError(f'Unknown family: {family}')


def load_state(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)


def compute_query_text_embedding(caption: str, device: torch.device) -> torch.Tensor:
    """Compute CLIP text embedding (512-dim) for a query caption using HF CLIP."""
    model_name = config.CLIP_MODEL_NAME if hasattr(config, 'CLIP_MODEL_NAME') else 'openai/clip-vit-base-patch32'
    try:
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name)
        clip_model.eval()
        # Prefer CPU for small inference to avoid MPS quirks
        inputs = tokenizer([caption], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            out = clip_model.text_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            txt = out.pooler_output
            if hasattr(clip_model, 'text_projection') and clip_model.text_projection is not None:
                txt = clip_model.text_projection(txt)
        txt = F.normalize(txt, p=2, dim=1)
        return txt.squeeze(0)
    except Exception as e:
        raise RuntimeError(f"Failed to compute text embedding for query caption: {e}")


def retrieve_top_k_images(model, loader: DataLoader, device: torch.device, query_emb: torch.Tensor, k: int) -> List[Tuple[int, float]]:
    model.eval()
    scores_ids: List[Tuple[float, int]] = []
    with torch.no_grad():
        q = query_emb.to(device).view(1, -1).t()
        for images, _txt_embeds, image_ids in loader:
            images = images.to(device)
            img_emb = F.normalize(model.encode_image(images), p=2, dim=1)
            sims = (img_emb @ q).squeeze(1).detach().cpu()
            for s, iid in zip(sims.tolist(), image_ids):
                scores_ids.append((s, int(iid)))
    # Get global top-k
    scores_ids.sort(key=lambda x: x[0], reverse=True)
    return [(iid, score) for score, iid in scores_ids[:k]]


def render_pdf(caption: str, topk: List[Tuple[int, float]], id_to_path: Dict[int, str], out_path: str):
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Caption: {caption}", fontsize=12)
    n = len(topk)
    cols = 5
    for i, (iid, score) in enumerate(topk):
        ax = plt.subplot(1, cols, i+1)
        path = id_to_path.get(iid, '')
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (256, 256), color=(0, 0, 0))
        ax.imshow(img)
        ax.set_title(f"ID {iid}\n{score:.3f}", fontsize=8)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption', type=str, required=True)
    parser.add_argument('--families', type=str, nargs='+', default=['baseline','modified','modified2','modified3'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--subset_percentage', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=5)
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

    # Data
    num_workers = args.num_workers if args.num_workers is not None else config.NUM_WORKERS
    loader, id_to_path = build_val_loader(args.batch_size, num_workers, args.pin_memory, args.subset_percentage)

    # Query embedding (CPU ok)
    query_emb = compute_query_text_embedding(args.caption, device=torch.device('cpu'))

    # Checkpoints
    groups = list_checkpoints(config.CHECKPOINTS_DIR)
    slug = slugify(args.caption)

    for family in args.families:
        flist = groups.get(family, [])
        if not flist:
            print(f"[skip] No checkpoints found for {family}")
            continue
        ckpt_name = pick_best_or_latest(flist)
        ckpt_path = os.path.join(config.CHECKPOINTS_DIR, ckpt_name)
        print(f"Evaluating {family} -> {ckpt_name}")

        # Model
        model = create_model_for_family(family, device)
        load_state(model, ckpt_path)

        # Retrieve
        topk = retrieve_top_k_images(model, loader, device, query_emb, args.top_k)

        # Render PDF
        out_dir = os.path.join(config.TRAIN_DIR, 'eval_results', family)
        out_pdf = os.path.join(out_dir, f"{family}_top{args.top_k}_{slug}.pdf")
        render_pdf(args.caption, topk, id_to_path, out_pdf)
        print(f"✓ Wrote {out_pdf}")


if __name__ == '__main__':
    main()
