"""
Evaluation script for Image-to-Text Translation model
"""
import os
import torch
import warnings
from transformers import AutoTokenizer

from src.config import CONFIG, device
from src.models import ResNetEncoderCustomDecoder
from src.dataset import create_data_loaders
from utils.metrics import evaluate_bleu, evaluate_with_examples

warnings.filterwarnings("ignore")


def load_model(checkpoint_path, tokenizer, config):
    """Load model from checkpoint."""
    print(f"📦 Loading model from: {checkpoint_path}")
    
    model = ResNetEncoderCustomDecoder(
        encoder_dim=config.encoder_dim,
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        pretrained=False  # No need for pretrained weights during eval
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    bleu = checkpoint.get('bleu', 0)
    epoch = checkpoint.get('epoch', 0)
    print(f"✅ Model loaded! Epoch: {epoch} | BLEU: {bleu:.4f}\n")
    
    return model


def main():
    """Main evaluation function."""
    print("\n" + "="*70)
    print("🔬 MODEL EVALUATION")
    print("="*70)
    print(f"🖥️  Device: {device}\n")
    
    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_name)
    print(f"✅ Tokenizer loaded: {tokenizer.vocab_size} tokens\n")
    
    # Create data loaders
    print("📦 Creating data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(CONFIG, tokenizer)
    print(f"✅ Validation set: {len(val_dataset)} samples\n")
    
    # Load model
    if not os.path.exists(CONFIG.checkpoint_path):
        print(f"❌ Checkpoint not found: {CONFIG.checkpoint_path}")
        return
    
    model = load_model(CONFIG.checkpoint_path, tokenizer, CONFIG)
    
    # Evaluate on full validation set
    print("="*70)
    print("🔍 FULL VALIDATION SET EVALUATION")
    print("="*70)
    
    final_bleu = evaluate_bleu(
        model, val_dataset, tokenizer, CONFIG, 
        num_samples=len(val_dataset)
    )
    
    # Show example predictions
    evaluate_with_examples(model, val_dataset, tokenizer, CONFIG, num_examples=5)
    
    print("\n" + "="*70)
    print(f"✅ EVALUATION COMPLETED!")
    print(f"🏆 Final BLEU-4: {final_bleu:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
