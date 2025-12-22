"""
Training script for Image-to-Text Translation model
"""
import os
import time
import torch
import torch.nn as nn
import warnings
from transformers import AutoTokenizer

from src.config import CONFIG, device
from src.models import ResNetEncoderCustomDecoder, get_model_info
from src.dataset import create_data_loaders
from utils.metrics import evaluate_bleu

warnings.filterwarnings("ignore")


def get_warmup_lr(epoch, base_lr, warmup_epochs):
    """Calculate learning rate with warmup."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def get_optimizer_and_scheduler(model, config, encoder_params, decoder_params):
    """Create optimizer and scheduler."""
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 0.0},  # Initially frozen
        {'params': decoder_params, 'lr': config.decoder_lr}
    ], weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    return optimizer, scheduler


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()
    optimizer.zero_grad()
    
    for i, (imgs, captions, lengths, raw_texts) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        
        # Prepare teacher forcing inputs/outputs
        tgt_input = captions[:, :-1]
        tgt_output = captions[:, 1:]
        
        # Forward pass
        if config.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(imgs, tgt_input)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)), 
                    tgt_output.reshape(-1)
                )
                loss = loss / config.gradient_accumulation_steps
        else:
            outputs = model(imgs, tgt_input)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                tgt_output.reshape(-1)
            )
            loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        if config.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (i + 1) % config.gradient_accumulation_steps == 0:
            if config.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        
        # Print progress
        if i % 200 == 0:
            print(f"  Step {i}/{len(train_loader)} | Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_time
    
    return avg_loss, elapsed


def adjust_learning_rates(optimizer, epoch, config, encoder_params):
    """Adjust learning rates based on training phase."""
    current_phase = 1
    
    if epoch < config.phase1_epochs:
        # Phase 1: Frozen encoder with warmup
        current_phase = 1
        for param in encoder_params:
            param.requires_grad = False
        
        warmup_progress = epoch - config.start_epoch
        dec_lr = get_warmup_lr(warmup_progress, config.decoder_lr, config.warmup_epochs)
        optimizer.param_groups[0]['lr'] = 0.0
        optimizer.param_groups[1]['lr'] = dec_lr
        phase_name = f"PHASE 1 (Frozen Encoder) - Warmup {warmup_progress+1}/{config.warmup_epochs}"
        
    elif epoch < config.phase2_epochs:
        # Phase 2: Unfreeze encoder with moderate LR
        current_phase = 2
        if epoch == config.phase1_epochs:
            print("\n" + "="*70)
            print("🔓 PHASE 2: Unfreezing encoder (Higher LR)")
            print("="*70)
            for param in encoder_params:
                param.requires_grad = True
        
        optimizer.param_groups[0]['lr'] = config.encoder_lr * 2.5
        optimizer.param_groups[1]['lr'] = config.decoder_lr
        phase_name = "PHASE 2 (Moderate Encoder LR)"
        
    else:
        # Phase 3: Normal dual LR
        current_phase = 3
        if epoch == config.phase2_epochs:
            print("\n" + "="*70)
            print("⚡ PHASE 3: Normal dual LR")
            print("="*70)
        
        optimizer.param_groups[0]['lr'] = config.encoder_lr
        optimizer.param_groups[1]['lr'] = config.decoder_lr
        phase_name = "PHASE 3 (Normal Dual LR)"
    
    return current_phase, phase_name


def save_checkpoint(model, optimizer, epoch, bleu, config, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bleu': bleu,
        'config': config.__dict__ if hasattr(config, '__dict__') else config
    }
    filepath = os.path.join(config.output_dir, filename)
    torch.save(checkpoint, filepath)
    return filepath


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("🛡️ IMAGE-TO-TEXT TRANSLATION TRAINING")
    print("="*70)
    print(f"🖥️  Device: {device}")
    print(CONFIG)
    
    # Create output directory
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_name)
    print(f"✅ Tokenizer: {tokenizer.vocab_size} tokens")
    
    # Create data loaders
    print("\n📦 Creating data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(CONFIG, tokenizer)
    print("✅ Data loaders ready")
    
    # Build model
    print("\n🔨 Building model...")
    model = ResNetEncoderCustomDecoder(
        encoder_dim=CONFIG.encoder_dim,
        vocab_size=tokenizer.vocab_size,
        embed_dim=CONFIG.embed_dim,
        n_heads=CONFIG.n_heads,
        n_layers=CONFIG.n_layers,
        ffn_dim=CONFIG.ffn_dim,
        dropout=CONFIG.dropout,
        pretrained=True
    ).to(device)
    
    model_info = get_model_info(model)
    print(f"✅ Model built!")
    print(f"   Total params: {model_info['total_params']:,}")
    print(f"   Trainable params: {model_info['trainable_params']:,}")
    
    # Load checkpoint if exists
    start_epoch = CONFIG.start_epoch
    best_bleu = 0.0
    
    if os.path.exists(CONFIG.checkpoint_path):
        print(f"\n📦 Loading checkpoint from {CONFIG.checkpoint_path}")
        checkpoint = torch.load(CONFIG.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_bleu = checkpoint.get('bleu', 0)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✅ Checkpoint loaded! Epoch: {start_epoch} | BLEU: {best_bleu:.4f}")
    
    # Get parameter groups
    encoder_params = list(model.get_encoder_parameters())
    decoder_params = list(model.get_decoder_parameters())
    
    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, CONFIG, encoder_params, decoder_params)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=CONFIG.label_smoothing
    )
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if CONFIG.use_amp else None
    
    # Training loop
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    no_improvement = 0
    
    for epoch in range(start_epoch, CONFIG.epochs):
        # Adjust learning rates
        current_phase, phase_name = adjust_learning_rates(optimizer, epoch, CONFIG, encoder_params)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG.epochs} - {phase_name}")
        print(f"LR: Enc={optimizer.param_groups[0]['lr']:.2e}, Dec={optimizer.param_groups[1]['lr']:.2e}")
        print(f"{'='*70}")
        
        # Train one epoch
        avg_loss, elapsed = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, CONFIG, epoch
        )
        print(f"\n✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {elapsed/60:.1f}m")
        
        # Validation
        current_bleu = evaluate_bleu(model, val_dataset, tokenizer, CONFIG, num_samples=CONFIG.eval_samples)
        scheduler.step(avg_loss)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save best model
        if current_bleu > best_bleu:
            improvement = (current_bleu - best_bleu) * 100
            best_bleu = current_bleu
            no_improvement = 0
            
            save_checkpoint(model, optimizer, epoch, best_bleu, CONFIG, "best_model.pth")
            print(f"💾 BEST! BLEU: {best_bleu:.4f} (↑ {improvement:.2f} pts)\n")
        else:
            no_improvement += 1
            print(f"⚠️ No improvement: {no_improvement} epoch(s)\n")
        
        # Early stopping (Phase 3 only)
        if current_phase == 3 and no_improvement >= CONFIG.early_stopping_patience:
            print(f"🛑 Early stopping triggered after {no_improvement} epochs without improvement")
            break
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, current_bleu, CONFIG, f"checkpoint_epoch_{epoch+1}.pth")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print(f"🏆 Best BLEU: {best_bleu:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
