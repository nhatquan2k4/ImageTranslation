"""
Inference script for single image prediction
"""
import os
import torch
import numpy as np
from PIL import Image
import warnings
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import CONFIG, device
from src.models import ResNetEncoderCustomDecoder
from utils.beam_search import beam_search, greedy_decode

warnings.filterwarnings("ignore")


class ImagePreprocessor:
    """Preprocessor for inference images."""
    
    def __init__(self, target_h=224, min_w=224, max_w=1890):
        self.target_h = target_h
        self.min_w = min_w
        self.max_w = max_w
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def __call__(self, img_path):
        """
        Load and preprocess image.
        
        Args:
            img_path: Path to image file
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize
        w, h = image.size
        scale = self.target_h / h
        new_w = int(w * scale)
        new_w = max(self.min_w, min(new_w, self.max_w))
        image = image.resize((new_w, self.target_h), Image.Resampling.BICUBIC)
        
        # Transform
        img_array = np.array(image)
        transformed = self.transform(image=img_array)["image"]
        
        return transformed


def load_model_for_inference(checkpoint_path, tokenizer, config):
    """Load model for inference."""
    print(f"📦 Loading model from: {checkpoint_path}")
    
    model = ResNetEncoderCustomDecoder(
        encoder_dim=config.encoder_dim,
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        pretrained=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded!\n")
    return model


def predict_single_image(model, image_path, tokenizer, preprocessor, config, use_beam_search=True):
    """
    Predict caption for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        tokenizer: Tokenizer
        preprocessor: Image preprocessor
        config: Configuration
        use_beam_search: Whether to use beam search (True) or greedy decoding (False)
    
    Returns:
        Predicted text
    """
    # Preprocess image
    image_tensor = preprocessor(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            if use_beam_search:
                pred_tokens = beam_search(
                    model, image_tensor, tokenizer,
                    beam_size=config.beam_size,
                    max_len=config.max_length,
                    length_penalty=config.length_penalty,
                    repetition_penalty=config.repetition_penalty
                )
            else:
                pred_tokens = greedy_decode(
                    model, image_tensor, tokenizer,
                    max_len=config.max_length
                )
    
    # Decode tokens
    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    return pred_text


def predict_batch(model, image_paths, tokenizer, preprocessor, config):
    """
    Predict captions for multiple images.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        tokenizer: Tokenizer
        preprocessor: Image preprocessor
        config: Configuration
    
    Returns:
        List of predicted texts
    """
    predictions = []
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"⚠️ File not found: {img_path}")
            predictions.append(None)
            continue
        
        pred_text = predict_single_image(
            model, img_path, tokenizer, preprocessor, config
        )
        predictions.append(pred_text)
    
    return predictions


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Image-to-Text Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of beam search")
    parser.add_argument("--show-image", action="store_true", help="Display input image")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔍 IMAGE-TO-TEXT INFERENCE")
    print("="*70)
    print(f"🖥️  Device: {device}\n")
    
    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_name)
    print(f"✅ Tokenizer loaded\n")
    
    # Load model
    checkpoint_path = args.checkpoint if args.checkpoint else CONFIG.checkpoint_path
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    model = load_model_for_inference(checkpoint_path, tokenizer, CONFIG)
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_h=CONFIG.target_h,
        min_w=CONFIG.min_w,
        max_w=CONFIG.max_w
    )
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    print(f"📷 Image: {args.image}")
    
    # Show image if requested
    if args.show_image:
        try:
            import matplotlib.pyplot as plt
            img = Image.open(args.image)
            plt.figure(figsize=(12, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Input Image", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("⚠️ Matplotlib not available for image display")
    
    # Predict
    print("\n" + "-"*70)
    print("🤖 Generating caption...")
    print("-"*70)
    
    prediction = predict_single_image(
        model, args.image, tokenizer, preprocessor, CONFIG,
        use_beam_search=not args.greedy
    )
    
    print(f"\n📝 Predicted Caption:")
    print(f"   {prediction}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
