"""
Dataset classes and data transforms for Image-to-Text Translation
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DynamicWidthTransform:
    """
    Transform that resizes images to fixed height while maintaining aspect ratio.
    Width is constrained between min_w and max_w.
    """
    
    def __init__(self, target_h, min_w, max_w, is_train=True):
        """
        Args:
            target_h: Target height for resized images
            min_w: Minimum width
            max_w: Maximum width
            is_train: Whether to apply training augmentations
        """
        self.target_h = target_h
        self.min_w = min_w
        self.max_w = max_w
        
        if is_train:
            self.transform = A.Compose([
                A.Rotate(limit=5, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    def __call__(self, img_pil):
        """
        Args:
            img_pil: PIL Image
        Returns:
            Tuple of (transformed_tensor, new_width)
        """
        w, h = img_pil.size
        scale = self.target_h / h
        new_w = int(w * scale)
        new_w = max(self.min_w, min(new_w, self.max_w))
        
        img_pil = img_pil.resize((new_w, self.target_h), Image.Resampling.BICUBIC)
        img_array = np.array(img_pil)
        transformed = self.transform(image=img_array)["image"]
        
        return transformed, new_w


class ImageTextDataset(Dataset):
    """
    Dataset for Image-to-Text translation.
    Loads images and corresponding text captions from JSON file.
    """
    
    def __init__(self, root_dir, json_file, tokenizer, transform=None, max_samples=None):
        """
        Args:
            root_dir: Root directory containing data
            json_file: Path to JSON file with annotations (relative to root_dir)
            tokenizer: Tokenizer for text processing
            transform: Image transformation function
            max_samples: Maximum number of samples to load (None for all)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Load annotations
        full_path = os.path.join(root_dir, json_file)
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)["data"]
        
        self.data = data[:max_samples] if max_samples else data
        
        # Determine correct image base path
        if len(self.data) > 0:
            sample = self.data[0]
            img_rel_path = sample.get('image_path', '').replace('\\', '/')
            full_img_path = os.path.join(root_dir, img_rel_path)
            
            if os.path.exists(full_img_path):
                self.image_base = root_dir
            else:
                parent_dir = os.path.dirname(root_dir)
                alt_path = os.path.join(parent_dir, img_rel_path)
                self.image_base = parent_dir if os.path.exists(alt_path) else root_dir
        else:
            self.image_base = root_dir
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            Tuple of (image_tensor, token_ids, target_text, image_width)
        """
        item = self.data[idx]
        
        # Load image
        img_rel_path = item['image_path'].replace('\\', '/')
        img_path = os.path.join(self.image_base, img_rel_path)
        image = Image.open(img_path).convert('RGB')
        
        # Transform image
        if self.transform:
            image, img_width = self.transform(image)
        else:
            img_width = image.size[0]
        
        # Tokenize text
        target_text = item['target_text']
        encoding = self.tokenizer(target_text, add_special_tokens=True, return_tensors='pt')
        ids = encoding['input_ids'].squeeze(0)
        
        return image, ids, target_text, img_width
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch, tokenizer, max_w_batch=1280, target_h=224):
    """
    Custom collate function for batching samples with variable-width images.
    
    Args:
        batch: List of samples from dataset
        tokenizer: Tokenizer for padding
        max_w_batch: Maximum width for batch
        target_h: Target height
    Returns:
        Tuple of (padded_images, padded_targets, lengths, raw_texts)
    """
    # Sort by caption length (longest first)
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    
    images, captions, raw_texts, img_widths = zip(*batch)
    
    # Pad images to same width
    max_width_in_batch = min(max(img_widths), max_w_batch)
    batch_size = len(images)
    padded_images = torch.zeros(batch_size, 3, target_h, max_width_in_batch)
    
    for i, (img, w) in enumerate(zip(images, img_widths)):
        actual_w = min(w, max_width_in_batch)
        padded_images[i, :, :, :actual_w] = img[:, :, :actual_w]
    
    # Pad captions
    lengths = [len(cap) for cap in captions]
    targets = torch.full((len(captions), max(lengths)), tokenizer.pad_token_id, dtype=torch.long)
    
    for i, cap in enumerate(captions):
        targets[i, :lengths[i]] = cap[:lengths[i]]
    
    return padded_images, targets, lengths, raw_texts


def create_data_loaders(config, tokenizer):
    """
    Create train and validation data loaders.
    
    Args:
        config: Configuration object
        tokenizer: Tokenizer instance
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    import random
    
    # Load all data
    full_json_path = os.path.join(config.data_root, config.json_file)
    with open(full_json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)["data"][:config.max_samples]
    
    # Split data
    total_samples = len(all_data)
    train_size = int(config.train_ratio * total_samples)
    
    indices = list(range(total_samples))
    random.seed(config.random_seed)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"📊 Dataset split: Train={train_size}, Val={len(val_indices)}")
    
    # Create transforms
    train_transform = DynamicWidthTransform(
        config.target_h, config.min_w, config.max_w, is_train=True
    )
    val_transform = DynamicWidthTransform(
        config.target_h, config.min_w, config.max_w, is_train=False
    )
    
    # Create datasets
    train_dataset_full = ImageTextDataset(
        config.data_root, config.json_file, tokenizer, 
        train_transform, config.max_samples
    )
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    
    val_dataset_full = ImageTextDataset(
        config.data_root, config.json_file, tokenizer,
        val_transform, config.max_samples
    )
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # Create custom collate function with config parameters
    def custom_collate(batch):
        return collate_fn(batch, tokenizer, config.max_w_batch, config.target_h)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset
