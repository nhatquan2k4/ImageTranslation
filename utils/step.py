import torch
from utils.mask import create_mask

def step(model, opt, batch, criterion, trg_pad, device):
    """
    Training step for image-to-translation model
    
    Args:
        model: Image-to-translation transformer model
        opt: Optimizer with learning rate scheduler
        batch: Tuple of (images, targets) from DataLoader
        criterion: Loss function (e.g., LabelSmoothingLoss)
        trg_pad: Padding token index for target vocabulary
        device: torch.device
    
    Returns:
        loss.item(): Training loss value
    """
    model.train()
    
    # Unpack batch
    src_img, trg = batch
    
    # Prepare decoder input and target
    trg_input = trg[:, :-1]  # Remove last token for decoder input
    trg_target = trg[:, 1:]  # Remove first token for target (shifted by 1)
    
    # Create target mask để prevent attention to future tokens
    trg_mask = create_mask(trg_input, trg_pad, True, device)
    
    # Forward pass
    preds = model(src_img, trg_input, trg_mask)
    
    # Reshape predictions và targets cho loss computation
    preds = preds.contiguous().view(-1, preds.size(-1))  # [batch*seq_len, vocab_size]
    targets = trg_target.contiguous().view(-1)  # [batch*seq_len]
    
    # Compute loss
    opt.zero_grad()
    loss = criterion(preds, targets)
    loss.backward()
    
    # Gradient clipping để prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update parameters
    opt.step_and_update_lr()
    
    return loss.item()