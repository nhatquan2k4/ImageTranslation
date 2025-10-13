"""
Example script to test the image-to-translation project
"""

import os
import json
import torch
from PIL import Image
from modules.transformer import Transformer
from utils.tokenizer import tokenizer
from utils.translator import translate
from utils.image_processor import ImageProcessor

def test_project():
    """
    Test function for the image-to-translation project
    """
    print("=== Image-to-Translation Project Test ===")
    
    # Check if required files exist
    config_path = "config/config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    # Load config
    with open(config_path) as f:
        cfg = json.load(f)
    print("✅ Config loaded successfully")
    print(f"Config: {json.dumps(cfg, indent=2)}")
    
    # Test tokenizer
    try:
        print("\n--- Testing Tokenizer ---")
        vn_tokenizer = tokenizer('vi')
        test_text = "Xin chào thế giới"
        tokens = vn_tokenizer.tokenize(test_text)
        print(f"Input: {test_text}")
        print(f"Tokens: {tokens}")
        print("✅ Vietnamese tokenizer works")
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        print("💡 Install underthesea: pip install underthesea")
    
    # Test image processor
    try:
        print("\n--- Testing Image Processor ---")
        image_processor = ImageProcessor(image_size=cfg['image_size'])
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='white')
        processed = image_processor.process_image(dummy_image, is_training=False)
        print(f"Processed image shape: {processed.shape}")
        print("✅ Image processor works")
    except Exception as e:
        print(f"❌ Image processor error: {e}")
    
    # Test model initialization
    try:
        print("\n--- Testing Model ---")
        # Create dummy vocabulary
        dummy_vocab = {
            '<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3,
            'hello': 4, 'world': 5, 'xin': 6, 'chào': 7
        }
        
        model = Transformer(
            trg_vocab=len(dummy_vocab),
            d_model=cfg['d_model'],
            n=cfg['n_layers'],
            heads=cfg['heads'],
            dropout=cfg['dropout'],
            image_size=cfg['image_size'],
            vision_hidden_dim=cfg.get('vision_hidden_dim', 768),
            vision_layers=cfg.get('vision_layers', 6),
            vision_heads=cfg.get('vision_heads', 8),
            patch_size=cfg.get('patch_size', 16)
        )
        
        # Test forward pass
        dummy_image = torch.randn(1, 3, cfg['image_size'], cfg['image_size'])
        dummy_target = torch.tensor([[2, 6, 7, 3]])  # <sos> xin chào <eos>
        
        with torch.no_grad():
            output = model(dummy_image, dummy_target[:, :-1], None)
        
        print(f"Model output shape: {output.shape}")
        print("✅ Model initialization and forward pass successful")
        
    except Exception as e:
        print(f"❌ Model error: {e}")
    
    print("\n=== Test Summary ===")
    print("✅ Project structure is ready for image-to-translation")
    print("📝 Next steps:")
    print("   1. Prepare your image-to-translation dataset")
    print("   2. Install required packages: pip install underthesea torch torchvision torchtext")
    print("   3. Run training: python train.py -d /path/to/data -m /path/to/model")
    print("   4. Run inference: python inference.py -p /path/to/image.jpg -m /path/to/model")

if __name__ == "__main__":
    test_project()
