from utils.init_matrix import create_init_matrix
from utils.mask import create_mask
import torch
import torch.nn.functional as F
import math

def find_best_k(k_res, last_word_out, current_log_scores, current_length, k):
    """
    Find the k best candidates at current step
    """
    probs, last_word_idx = last_word_out.data.topk(k)

    log_scores = torch.log(probs + 1e-12)  # Avoid log(0)
    log_scores = log_scores.to(current_log_scores.device)
    log_scores = log_scores + current_log_scores  # broadcast addition

    log_scores = log_scores.view(-1)
    k_probs, k_idx = log_scores.topk(k)
    row = k_idx // k
    col = k_idx % k
    new_log_scores = k_probs.unsqueeze(1)

    k_res[:, :current_length] = k_res[row, :current_length]
    k_res[:, current_length] = last_word_idx[row, col]

    return k_res, new_log_scores


def beam_search(src_img, model, vocab, device, k, max_strlen):
    """
    Beam search for image-to-translation
    
    Args:
        src_img: Input image tensor [1, 3, H, W]
        model: Trained image-to-translation model
        vocab: Target vocabulary (Vietnamese)
        device: torch.device
        k: Beam width
        max_strlen: Maximum sequence length
    
    Returns:
        tuple: (best_sequence, length)
    """
    # Initialize beam search matrix
    e_out, k_res, log_scores = create_init_matrix(src_img, model, vocab, device, k, max_strlen)

    # Get special token indices
    if hasattr(vocab, 'get_stoi'):
        # torchtext vocab object
        eos_token = vocab['<eos>']
        pad_token = vocab['<pad>']
    else:
        # Dictionary vocab
        eos_token = vocab['<eos>']
        pad_token = vocab['<pad>']
    
    src_mask = None  # No source mask needed for images
    best_idx = 0  # Default beam 0

    for i in range(2, max_strlen):
        # Create target mask to prevent attention to future tokens
        trg_mask = create_mask(k_res[:, :i], pad_token, True, device)
        
        # Forward pass through decoder
        preds = model(src_img.expand(k, -1, -1, -1), k_res[:, :i], trg_mask)
        
        # Get probabilities for next token
        out = F.softmax(preds, dim=-1)
        out = out[:, -1]  # Get last token predictions

        # Find k best candidates
        k_res, log_scores = find_best_k(k_res, out, log_scores, i, k)

        # Check for EOS tokens
        sentence_lengths = torch.zeros(k, dtype=torch.long).to(device)
        end_positions = (k_res == eos_token).nonzero(as_tuple=False)
        
        for pos in end_positions:
            sentence_idx = pos[0].item()
            sentence_len = pos[1].item()
            if sentence_lengths[sentence_idx] == 0:  # First EOS for this sentence
                sentence_lengths[sentence_idx] = sentence_len

        # Check if all beams have finished
        n_finished = (sentence_lengths > 0).sum().item()
        if n_finished == k:
            # Apply length normalization để prefer longer sequences
            alpha = 0.7
            denominator = torch.where(
                sentence_lengths > 0, 
                sentence_lengths.float(), 
                torch.full_like(sentence_lengths, max_strlen, dtype=torch.float)
            )
            normalized_scores = log_scores.view(-1) / (denominator ** alpha)
            best_idx = torch.argmax(normalized_scores).item()
            break

    # Extract best sequence
    best_sentence = k_res[best_idx].view(-1)
    eos_positions = (best_sentence == eos_token).nonzero(as_tuple=True)[0]
    length = eos_positions[0].item() if len(eos_positions) > 0 else max_strlen

    return best_sentence[:length].cpu().numpy(), length
