"""
Beam search implementation for text generation
"""
import math
import torch


def beam_search(model, image, tokenizer, beam_size=5, max_len=70,
                length_penalty=0.6, repetition_penalty=1.2):
    """
    Beam search for generating text from images.
    
    Args:
        model: The trained model
        image: Input image tensor of shape (1, 3, H, W)
        tokenizer: Tokenizer for encoding/decoding
        beam_size: Number of beams
        max_len: Maximum generation length
        length_penalty: Length normalization factor
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        List of token IDs for the best sequence
    """
    model.eval()
    
    with torch.no_grad():
        # Encode image
        features = model.encoder(image)
        features = model.adaptive_pool(features)
        B, C, _, W = features.shape
        memory = features.squeeze(2).permute(0, 2, 1)
        memory = model.enc_project(memory)
    
    # Initialize beams with BOS token
    beams = [([tokenizer.bos_token_id], 0.0)]
    
    for step in range(max_len):
        new_beams = []
        
        for seq, score in beams:
            # If sequence already ended, keep it
            if seq[-1] == tokenizer.eos_token_id:
                new_beams.append((seq, score))
                continue
            
            # Prepare input
            tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(image.device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(image.device)
            
            # Embed and encode
            tgt_emb = model.embedding(tgt_tensor) * math.sqrt(model.embed_dim)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            # Decode
            with torch.no_grad():
                output = model.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                output = model.fc_out(output)
                log_probs = torch.log_softmax(output[0, -1, :], dim=-1)
            
            # Apply repetition penalty
            for token_id in set(seq):
                if token_id not in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
                    log_probs[token_id] /= repetition_penalty
            
            # Get top-k candidates
            top_log_probs, top_indices = torch.topk(log_probs, beam_size)
            
            # Extend beams
            for log_prob, idx in zip(top_log_probs, top_indices):
                new_seq = seq + [idx.item()]
                new_score = score + log_prob.item()
                new_beams.append((new_seq, new_score))
        
        # Select top beams with length normalization
        beams = sorted(
            new_beams, 
            key=lambda x: x[1] / (len(x[0]) ** length_penalty), 
            reverse=True
        )[:beam_size]
        
        # Early stopping if all beams ended
        if all(seq[-1] == tokenizer.eos_token_id for seq, _ in beams):
            break
    
    # Return best sequence
    best_seq = max(beams, key=lambda x: x[1] / (len(x[0]) ** length_penalty))[0]
    return best_seq


def greedy_decode(model, image, tokenizer, max_len=70):
    """
    Greedy decoding for faster inference.
    
    Args:
        model: The trained model
        image: Input image tensor
        tokenizer: Tokenizer
        max_len: Maximum length
    
    Returns:
        List of token IDs
    """
    model.eval()
    
    with torch.no_grad():
        # Encode image
        features = model.encoder(image)
        features = model.adaptive_pool(features)
        memory = features.squeeze(2).permute(0, 2, 1)
        memory = model.enc_project(memory)
        
        # Start with BOS token
        tokens = [tokenizer.bos_token_id]
        
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tokens).unsqueeze(0).to(image.device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(image.device)
            
            tgt_emb = model.embedding(tgt_tensor) * math.sqrt(model.embed_dim)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            output = model.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            output = model.fc_out(output)
            
            # Get most likely token
            next_token = output[0, -1, :].argmax().item()
            tokens.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
        
        return tokens
