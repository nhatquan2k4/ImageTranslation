"""
Gradio Web Application for Image-to-Text Translation
"""
import os
import torch
import math
import numpy as np
import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer

from src.config import CONFIG, device
from src.models import ResNetEncoderCustomDecoder


# Global variables for model and tokenizer
model = None
tokenizer = None
preprocessor = None


class ImagePreprocessor:
    """Preprocessor for Gradio interface."""
    
    def __init__(self, target_h=224, min_w=224, max_w=1890):
        self.target_h = target_h
        self.min_w = min_w
        self.max_w = max_w
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def __call__(self, img_array):
        """Process image array from Gradio."""
        # Convert to PIL
        img_pil = Image.fromarray(img_array).convert('RGB')
        
        # Resize
        w, h = img_pil.size
        scale = self.target_h / h
        new_w = int(w * scale)
        new_w = max(self.min_w, min(new_w, self.max_w))
        img_resized = img_pil.resize((new_w, self.target_h), Image.Resampling.BICUBIC)
        
        # Transform
        img_array = np.array(img_resized)
        transformed = self.transform(image=img_array)["image"]
        
        return transformed


def load_model_and_tokenizer():
    """Load model and tokenizer."""
    global model, tokenizer, preprocessor
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_name)
    
    print("Loading model...")
    model = ResNetEncoderCustomDecoder(
        encoder_dim=CONFIG.encoder_dim,
        vocab_size=tokenizer.vocab_size,
        embed_dim=CONFIG.embed_dim,
        n_heads=CONFIG.n_heads,
        n_layers=CONFIG.n_layers,
        ffn_dim=CONFIG.ffn_dim,
        dropout=CONFIG.dropout,
        pretrained=False
    ).to(device)
    
    if os.path.exists(CONFIG.checkpoint_path):
        checkpoint = torch.load(CONFIG.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from checkpoint (BLEU: {checkpoint.get('bleu', 0):.4f})")
    else:
        print(f"Warning: Checkpoint not found at {CONFIG.checkpoint_path}")
    
    model.eval()
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(CONFIG.target_h, CONFIG.min_w, CONFIG.max_w)
    
    print("Ready!")


def predict_image(img_array):
    """
    Predict caption for uploaded image.
    
    Args:
        img_array: Numpy array from Gradio image input
    
    Returns:
        Predicted Vietnamese text
    """
    if img_array is None:
        return "❌ Vui lòng tải ảnh lên"
    
    if model is None or tokenizer is None:
        return "❌ Model chưa được load"
    
    try:
        # Preprocess image
        img_tensor = preprocessor(img_array)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Beam search
        with torch.no_grad():
            # Encode image
            features = model.encoder(img_tensor)
            features = model.adaptive_pool(features)
            memory = features.squeeze(2).permute(0, 2, 1)
            memory = model.enc_project(memory)
            
            # Beam search decode
            beams = [([tokenizer.bos_token_id], 0.0)]
            for _ in range(CONFIG.max_length):
                new_beams = []
                for seq, score in beams:
                    if seq[-1] == tokenizer.eos_token_id:
                        new_beams.append((seq, score))
                        continue
                    
                    tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
                    tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
                    tgt_emb = model.embedding(tgt_tensor) * math.sqrt(CONFIG.embed_dim)
                    tgt_emb = model.pos_encoder(tgt_emb)
                    
                    output = model.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    log_probs = torch.log_softmax(model.fc_out(output)[0, -1, :], dim=-1)
                    
                    # Apply repetition penalty
                    for token_id in set(seq):
                        if token_id not in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
                            log_probs[token_id] /= CONFIG.repetition_penalty
                    
                    top_probs, top_idxs = torch.topk(log_probs, CONFIG.beam_size)
                    for p, idx in zip(top_probs, top_idxs):
                        new_beams.append((seq + [idx.item()], score + p.item()))
                
                beams = sorted(
                    new_beams, 
                    key=lambda x: x[1] / (len(x[0]) ** CONFIG.length_penalty), 
                    reverse=True
                )[:CONFIG.beam_size]
                
                if all(s[-1] == tokenizer.eos_token_id for s, _ in beams):
                    break
        
        # Decode best sequence
        best_seq = beams[0][0]
        pred_text = tokenizer.decode(best_seq, skip_special_tokens=True)
        
        return pred_text
    
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"


def create_gradio_interface():
    """Create Gradio interface."""
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-text {
        font-size: 18px !important;
        font-weight: bold;
    }
    """
    
    # Create interface
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(
            label="📷 Tải ảnh lên",
            type="numpy"
        ),
        outputs=gr.Textbox(
            label="📝 Kết quả dịch (Tiếng Việt)",
            lines=3,
            elem_classes="output-text"
        ),
        title="🌍 Image-to-Text Translation (English → Vietnamese)",
        description="""
        ### Mô hình ResNet101-Transformer dịch ảnh chứa văn bản tiếng Anh sang tiếng Việt
        
        **Hướng dẫn sử dụng:**
        1. Tải ảnh chứa văn bản tiếng Anh lên
        2. Đợi mô hình xử lý (5-10 giây)
        3. Xem kết quả dịch sang tiếng Việt
        
        **Lưu ý:** Mô hình hoạt động tốt nhất với ảnh rõ nét, không bị mờ hoặc méo quá nhiều.
        """,
        examples=[
            # Add example image paths here if available
        ],
        theme=gr.themes.Soft(),
        css=custom_css,
        allow_flagging="never"
    )
    
    return interface


def main():
    """Main function to launch Gradio app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Image-to-Text Gradio App")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    
    args = parser.parse_args()
    
    # Update checkpoint path if provided
    if args.checkpoint:
        CONFIG.checkpoint_path = args.checkpoint
    
    print("\n" + "="*70)
    print("🚀 LAUNCHING GRADIO APP")
    print("="*70)
    print(f"Device: {device}")
    print(f"Checkpoint: {CONFIG.checkpoint_path}")
    print("="*70 + "\n")
    
    # Load model
    load_model_and_tokenizer()
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    print("\n" + "="*70)
    print("✅ Gradio app is ready!")
    print("="*70 + "\n")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
