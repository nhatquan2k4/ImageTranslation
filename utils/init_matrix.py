from utils.mask import create_mask
import torch
import torch.nn.functional as F
import math

def create_init_matrix(src_img, model, vocab, device, k, max_len):
    """
    Initialize beam search matrix for image-to-translation
    
    Args:
        src_img: Input image tensor [1, 3, H, W]
        model: Image-to-translation model
        vocab: Target vocabulary
        device: torch.device
        k: Beam width
        max_len: Maximum sequence length
    
    Returns:
        tuple: (vision_features, k_results, log_scores)
    """
    # Get start of sentence token
    init_token = vocab['<sos>']
    pad_token = vocab['<pad>']
    
    # Extract vision features once (will be reused for all beams)
    with torch.no_grad():
        vision_features = model.vision_encoder(src_img)  # [1, seq_len, d_model]
    
    # Initialize first target token (SOS)
    trg = torch.LongTensor([[init_token]]).to(device)  # [1, 1]
    
    # Create mask cho first token
    trg_mask = create_mask(trg, pad_token, True, device)
    
    # Get first predictions
    with torch.no_grad():
        preds = model(src_img, trg, trg_mask)  # [1, 1, vocab_size]
        out = F.softmax(preds, dim=-1)
        out = out[:, -1]  # Get last (first) token predictions [1, vocab_size]
    
    # Get top-K candidates for second token
    probs, idx = out.data.topk(k)  # [1, k]
    
    # Convert to log scores
    log_scores = torch.log(probs[0] + 1e-12)  # [k]
    log_scores = log_scores.unsqueeze(1)  # [k, 1]
    
    # Initialize result matrix
    k_res = torch.full((k, max_len), pad_token, dtype=torch.long).to(device)  # [k, max_len]
    k_res[:, 0] = init_token  # First token is always <sos>
    k_res[:, 1] = idx[0]      # Second tokens are top-K candidates
    
    # Expand vision features to k batch size for parallel processing
    vision_features_expanded = vision_features.expand(k, -1, -1)  # [k, seq_len, d_model]
    
    return vision_features_expanded, k_res, log_scores