
import os
import json
import random
from typing import Dict, List, Sequence, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config


def load_captions_from_json(captions_path: str) -> Dict[int, List[str]]:
    """Return a mapping of image_id -> list of captions from a COCO captions file."""
    with open(captions_path, "r", encoding="utf-8") as fp:
        coco_data = json.load(fp)

    captions = {}
    for annotation in coco_data.get("annotations", []):
        captions.setdefault(annotation["image_id"], []).append(annotation["caption"])
    return captions


def _ensure_tensor(embedding) -> torch.Tensor:
    """Coerce cached embedding entries (tensor, list, numpy) into torch.Tensor."""
    if isinstance(embedding, torch.Tensor):
        return embedding
    return torch.tensor(embedding)


def build_embedding_index(
    cached_records: Sequence[Dict[str, Union[int, torch.Tensor]]]
) -> Dict[int, List[torch.Tensor]]:
    """
    Convert cached embedding records into image_id -> [embeddings].

    The preprocessing script stores a list of dicts containing `image_id` and the
    corresponding CLIP text embedding. For training we want to sample a caption
    embedding per image on-the-fly to keep the dataset aligned with images.
    """
    embedding_index: Dict[int, List[torch.Tensor]] = {}
    for record in cached_records:
        image_id = int(record["image_id"])
        embedding_index.setdefault(image_id, []).append(_ensure_tensor(record["text_embedding"]))
    return embedding_index


class COCOCLIPDataset(Dataset):
    """
    PyTorch Dataset for COCO images backed by cached CLIP text embeddings.

    Each sample returns:
        - image tensor normalized with CLIP stats
        - paired text embedding (random caption per image for variety)
        - image id
    """

    def __init__(
        self,
        images_dir: str,
        captions_path: str,
        text_embeddings: Union[str, Sequence[Dict[str, Union[int, torch.Tensor]]], Dict[int, torch.Tensor]],
        transform=None,
    ):
        """
        Args:
            images_dir: Directory that contains the split images.
            captions_path: Path to the COCO captions json file for the split.
            text_embeddings: Either a path to the cached .pt file, the raw list produced
                             by `preprocess_data.py`, or a mapping image_id -> tensor(s).
            transform: Optional torchvision transform override.
        """
        self.images_dir = images_dir
        self.transform = transform or self._default_transform()

        # Load COCO metadata once for image filenames
        with open(captions_path, "r", encoding="utf-8") as fp:
            coco_data = json.load(fp)
        self.images = coco_data["images"]

        # Build caption lookup (useful for debugging or visualization)
        self.image_to_captions = load_captions_from_json(captions_path)

        # Normalize cached embeddings into image_id -> List[Tensor]
        if isinstance(text_embeddings, str):
            raw_embeddings = torch.load(text_embeddings, map_location="cpu")
        else:
            raw_embeddings = text_embeddings

        if isinstance(raw_embeddings, dict):
            # Allow dicts that map to a single tensor or list of tensors
            self.embedding_index = {
                int(img_id): value if isinstance(value, list) else [value]
                for img_id, value in raw_embeddings.items()
            }
        else:
            self.embedding_index = build_embedding_index(raw_embeddings)

        # Keep only images that have cached embeddings
        self.images = [img for img in self.images if img["id"] in self.embedding_index]
        if len(self.images) == 0:
            raise ValueError(
                "No overlapping images between captions file and cached embeddings. "
                "Ensure you ran `preprocess_data.py` on the same split."
            )

    def _default_transform(self):
        """CLIP preprocessing pipeline for images."""
        return transforms.Compose(
            [
                transforms.Resize((config.CLIP_IMAGE_SIZE, config.CLIP_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.CLIP_MEAN, std=config.CLIP_STD),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        filename = image_info["file_name"]

        image_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except (OSError, IOError) as exc:
            print(f"Warning: Could not load image {filename} (ID: {image_id}): {exc}")
            image = Image.new("RGB", (config.CLIP_IMAGE_SIZE, config.CLIP_IMAGE_SIZE), color=(0, 0, 0))
            image_tensor = self.transform(image)

        # Randomly pick one cached caption embedding for this image
        embeddings = self.embedding_index.get(image_id)
        if not embeddings:
            raise KeyError(f"No cached embedding found for image_id={image_id}")
        text_embedding = random.choice(embeddings)

        return image_tensor, text_embedding, image_id

    def get_raw_caption(self, image_id: int) -> str:
        """Return the first caption associated with image_id (useful for inspection)."""
        captions = self.image_to_captions.get(image_id, ["No caption available"])
        return captions[0]

    def get_image_path(self, idx: int) -> str:
        """Return absolute image path for visualization or debugging."""
        filename = self.images[idx]["file_name"]
        return os.path.join(self.images_dir, filename)
