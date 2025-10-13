import torch
from torchvision import transforms
from PIL import Image
import os

class ImageProcessor:
    def __init__(self, image_size=384):  # Increased size for better OCR
        self.image_size = image_size
        
        # Transform cho training (OCR-optimized augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Fixed size for consistency
            # More conservative augmentation for text preservation
            transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.05),
            # Remove horizontal flip - can damage text readability
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet norms
        ])
        
        # Transform cho validation/inference (không có augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def process(self, image_path, is_training=True):
        """
        Backward compatibility method
        """
        try:
            img = Image.open(image_path).convert('RGB')
            return self.process_image(img, is_training)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
            
    def process_image(self, image, is_training=True):
        """
        Process PIL Image object
        """
        try:
            if is_training:
                return self.train_transform(image)
            else:
                return self.val_transform(image)
        except Exception as e:
            print(f"Error processing image: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def process_for_inference(self, image_path):
        """
        Process image for inference
        """
        return self.process(image_path, is_training=False)