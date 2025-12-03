# ELEC475 Lab 4 Report (CLIP)

## 1. Title & Authors
- Project: Modified CLIP on COCO2014
- Team: <Names / Student IDs>

## 2. Objective
- Goal: <e.g., align image & text embeddings; improve retrieval/detection>
- Key Modifications: Refer to `models/clip_model_modified_2.py` / `_modified.py`.

## 3. Dataset
- Source: COCO 2014 (train/val) in `datasets/coco2014/`
- Tasks used: <captions / instances / keypoints>
- Splits: train = <N>, val = <N> (derive via loader)
- Preprocessing: See `datasets/preprocess_data.py`, `datasets/augmentation.py`.

## 4. Model Architecture
- Base: CLIP variant (File: `models/clip_model.py`)
- Changes: <layers frozen/unfrozen, added heads, embed dims>
- Loss functions: <contrastive / auxiliary>; see `train/train_clip.py`.

## 5. Training Configuration
- Hyperparams: batch=<>, epochs=<>, lr=<>, optimizer=<>, scheduler=<>
- Implemented in `train/train_clip.py`.
- Device / precision: <CPU/GPU / fp16?>
- Checkpoints: `train/checkpoints/`

## 6. Evaluation
- Metrics: <Recall@K / accuracy / loss curves>
- How computed: <brief method>; data loader in `datasets/dataloader.py`.
- Best checkpoint: `checkpoint_best.pt` (epoch <>)
 - Data usage: Baseline and v1 trained on 20% of COCO train; v2 trained on 100%.
 - Caption usage: Training uses 1 caption per image (out of 5 in COCO), which limits supervision and can reduce R@1.

## 7. Results (Quantitative)

Baseline (ResNet50 + 2-layer head)
- Hyperparams: batch=32, lr=5e-4, epochs=10, wd=0.01, device=MPS
- Data: 20% train split
- Final losses: train=0.2560, val=1.5638
- Recall (val, 8,100 samples):
	- i→t R@1=6.75%, R@5=20.00%, R@10=29.90%
	- t→i R@1=7.65%, R@5=23.14%, R@10=33.63%

Modified v1 (LN/BN + dropout head, partial unfreeze)
- Hyperparams: batch=32, lr=5e-4, epochs=10, wd=0.01, device=MPS
- Data: 20% train split
- Final losses: train=0.2168, val=1.4320 (best val=1.3728 @ epoch 5)
- Recall (val, 8,100 samples):
	- i→t R@1=10.17%, R@5=27.60%, R@10=39.40%
	- t→i R@1=11.10%, R@5=30.20%, R@10=41.75%

Modified v2 (enhanced 3-layer head + LN/Dropout, unfreeze last stage)
- Hyperparams: batch=32, lr=3e-4, epochs=10, wd=0.02, device=MPS
- Data: 100% train split
- Final losses: train=0.1468, val=1.0645
- Recall (val, 40,504 samples):
	- i→t R@1=7.25%, R@5=19.77%, R@10=28.25%
	- t→i R@1=9.20%, R@5=23.69%, R@10=32.78%

Modified v3 (ConvNeXt/EfficientNet options, residual LN head)
- Hyperparams: convnext_tiny preset: batch=32, lr=3e-4, epochs=10, wd=0.02, cosine+warmup, device=MPS
- Data: 20% train split
- Final losses: train=0.2048, val=1.1901
- Recall (val, 8,100 samples):
	- i→t R@1=13.86%, R@5=35.67%, R@10=48.95%
	- t→i R@1=17.30%, R@5=41.02%, R@10=54.46%

## 8. Results (Qualitative)
- Provide 2–4 image→top-5 captions (successes + 1–2 failures) using `val2014` and `datasets/cache/val_text_embeds.pt`.

## 9. Analysis (brief)
- v1 (20% data): regularized head + partial unfreezing yields clear R@K gains over baseline on the smaller split.
- v2 (100% data): deeper 3-layer head + last-stage unfreezing drives val loss down substantially on full val; R@1 modest due to single-caption training.
- v3 (ConvNeXt-Tiny, 20% data): strongest R@K so far on the 8,100-sample val (i→t R@1≈13.9%, t→i R@1≈17.3%), indicating backbone + residual LN head + dropout improve retrieval.
- τ behavior: converges/stabilizes early (≈0.03–0.04); clamping avoids collapse.
- Data supervision: using 1 caption (of 5) limits alignment diversity; multi-caption sampling likely boosts R@1.

## 10. Limitations
- Compute constraints.
- Dataset bias (object frequency skew).
- Model capacity / overfitting signs.

## 11. Ablations (optional)
- Not required; include one small change if available (e.g., dropout or unfreeze depth) and its Recall@K delta.

## 12. Implementation Notes
- Using cached text embeddings (`datasets/cache/*.pt`), text encoder skipped during training.
- Dataloader: `datasets/dataloader.py`; augmentation: `datasets/augmentation.py`.
 - Text embeddings preprocessing: `datasets/preprocess_data.py` tokenizes COCO captions and computes CLIP text embeddings, saved as `.pt` for train/val; these are loaded and L2-normalized per batch in training.

## 13. Repro Steps
1. `pip install -r requirements.txt`
2. Prepare COCO + cached embeddings: `python datasets/preprocess_data.py`
3. Train baseline: `python train/train_clip.py`
4. Train v1: `python train/train_clip.py --modified --mod_config layer_norm`
5. Train v2: `python train/train_clip.py --epochs 10 --batch_size 32 --modified2 --mod_config best_combo_v2 --scheduler cosine --warmup_epochs 1 --lr 3e-4 --weight_decay 0.02`
6. Evaluate (auto): loss curves + Recall@K saved in `train/`
7. Train v3 (ConvNeXt-Tiny, 20%): `python train/train_clip.py --epochs 10 --batch_size 32 --modified3 --mod_config convnext_tiny_v3 --scheduler cosine --warmup_epochs 1 --backbone_lr_scale 0.2 --early_stop_patience 3 --lr 3e-4 --weight_decay 0.02 --train_percentage 20`
7. Evaluate checkpoints for the report: `python3 train/eval_checkpoints.py --device auto`
Notes:
- To use 20% of the dataset for baseline/v1 runs: add `--train_percentage 20`.
- v2 used full 100% data (omit the flag).
 - v3 shown results are on 20% data; if you run 100%, report exact numbers.

## 14. Future Work
- Multi-caption sampling per image; stronger backbone (ConvNeXt/ViT); longer training with cosine.

## 15. References
- CLIP original paper.
- COCO dataset paper.

## 16. Appendix
- Hyperparameter table; loss curves; extra retrieval examples.

---
Fill <> placeholders with actual numbers extracted from logs & loaders. Keep total length concise per lab rubric.
