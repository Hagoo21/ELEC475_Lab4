# ELEC475_Lab4

## Pip Installs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers
pip install matplotlib
pip install tqdm
pip install Pillow
should be all 


## Dataset setup

- Download or prepare the COCO 2014 dataset and place it under `datasets/coco2014/` within this repo. The expected structure is:
	- `datasets/coco2014/images/train2014/`
	- `datasets/coco2014/images/val2014/`
	- `datasets/coco2014/annotations/captions_train2014.json`
	- `datasets/coco2014/annotations/captions_val2014.json`
	- (optional for EDA) `datasets/coco2014/annotations/instances_train2014.json`, `instances_val2014.json`

Make sure `config.py` points to these paths (TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_CAPTIONS_PATH, VAL_CAPTIONS_PATH).

## Commands

- Refer to `train/train.txt` for training command examples and presets.
- Refer to `train/eval.txt` for evaluation commands

Note: All model weights (pretrained/trained) are already included under `train/checkpoints/`.

## Preprocessing

- Caption embedding caching script: `datasets/preprocess_data.py`
- Run:

```
python datasets/preprocess_data.py
```




