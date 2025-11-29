
class COCOCLIPDataset(Dataset):
    """
    PyTorch Dataset for COCO images with precomputed CLIP text embeddings.
    
    Returns:
        image: Normalized image tensor (3, 224, 224)
        text_embedding: Precomputed CLIP text embedding (512,)
        image_id: Image ID from COCO dataset
    """
    
    def __init__(
        self,
        images_dir: str,
        captions_path: str,
        text_embeddings: Dict[int, torch.Tensor],
        transform=None
    ):
        """
        Args:
            images_dir: Directory containing images
            captions_path: Path to captions.json
            text_embeddings: Dictionary mapping image_id to text embeddings
            transform: Optional image transforms
        """
        self.images_dir = images_dir
        self.text_embeddings = text_embeddings
        
        # Load image metadata
        with open(captions_path, 'r') as f:
            coco_data = json.load(f)
        self.images = coco_data['images']
        
        # Load captions for verification
        self.image_to_captions = load_captions_from_json(captions_path)
        
        # Set up transforms
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform
    
    def _default_transform(self):
        """
        Create default CLIP-style image preprocessing transform.
        
        Pipeline:
            1. Resize to 224x224
            2. Convert to tensor (scales to [0, 1])
            3. Normalize with CLIP statistics
        """
        return transforms.Compose([
            transforms.Resize((config.CLIP_IMAGE_SIZE, config.CLIP_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.CLIP_MEAN, std=config.CLIP_STD)
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        image_info = self.images[idx]
        image_id = image_info['id']
        filename = image_info['file_name']
        
        # Load and transform image
        image_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except (OSError, IOError) as e:
            # If image is corrupted and can't be loaded, create a black image as fallback
            print(f"Warning: Could not load image {filename} (ID: {image_id}): {e}")
            # Create a black image of the expected size (224x224 for CLIP)
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            image_tensor = self.transform(image)
        
        # Get precomputed text embedding
        text_embedding = self.text_embeddings[image_id]
        
        return image_tensor, text_embedding, image_id
    
    def get_raw_caption(self, image_id: int) -> str:
        """Get the raw caption for an image (before tokenization)."""
        captions = self.image_to_captions.get(image_id, ["No caption available"])
        return captions[0]
    
    def get_image_path(self, idx: int) -> str:
        """Get the file path for an image."""
        image_info = self.images[idx]
        filename = image_info['file_name']
        return os.path.join(self.images_dir, filename)
