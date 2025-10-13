import torch
from utils.mask import create_mask
import numpy as np

def validiate(model, valid_iter, criterion, trg_pad, device):
    """
    Validation function for image-to-translation model
    
    Args:
        model: Image-to-translation transformer model
        valid_iter: Validation DataLoader
        criterion: Loss function
        trg_pad: Padding token index for target vocabulary
        device: torch.device
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    
    total_loss = []
    total_samples = 0
    
    with torch.no_grad():
        for batch in valid_iter:
            # Unpack batch
            src_img, trg = batch
            
            # Prepare decoder input and target
            trg_input = trg[:, :-1]  # Remove last token for decoder input
            trg_target = trg[:, 1:]  # Remove first token for target (shifted by 1)
            
            # Create target mask
            trg_mask = create_mask(trg_input, trg_pad, True, device)
            
            # Forward pass
            preds = model(src_img, trg_input, trg_mask)
            
            # Reshape predictions và targets
            preds = preds.contiguous().view(-1, preds.size(-1))
            targets = trg_target.contiguous().view(-1)
            
            # Compute loss
            loss = criterion(preds, targets)
            
            # Accumulate loss (weighted by batch size)
            batch_size = src_img.size(0)
            total_loss.append(loss.item() * batch_size)
            total_samples += batch_size
    
    # Return average loss
    if total_samples > 0:
        return sum(total_loss) / total_samples
    else:
        return float('inf')