import os
import torch
import json
import glob
from collections import OrderedDict

class CheckpointManager:
    def __init__(self, model_folder, max_checkpoints=3):
        self.model_folder = model_folder
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = os.path.join(model_folder, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, epoch, step, loss, vocab, config):
        """
        Save training checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch:03d}_step_{step:06d}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        torch.save(vocab, os.path.join(self.checkpoint_dir, f'vocab_epoch_{epoch:03d}.pth'))
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """
        Load training checkpoint
        """
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        loss = checkpoint.get('loss', float('inf'))
        
        print(f"✅ Resumed from epoch {epoch}, step {step}, loss {loss:.4f}")
        
        return epoch, step, loss
    
    def find_latest_checkpoint(self):
        """
        Find the latest checkpoint file
        """
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt')
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return None
            
        # Sort by modification time (latest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    
    def _cleanup_checkpoints(self):
        """
        Keep only the latest N checkpoints
        """
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt')
        checkpoints = glob.glob(pattern)
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time (latest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoints[self.max_checkpoints:]:
            try:
                os.remove(old_checkpoint)
                # Also remove corresponding vocab file
                epoch = self._extract_epoch_from_path(old_checkpoint)
                vocab_file = os.path.join(self.checkpoint_dir, f'vocab_epoch_{epoch:03d}.pth')
                if os.path.exists(vocab_file):
                    os.remove(vocab_file)
                print(f"🗑️ Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                print(f"⚠️ Failed to remove {old_checkpoint}: {e}")
    
    def _extract_epoch_from_path(self, checkpoint_path):
        """
        Extract epoch number from checkpoint path
        """
        basename = os.path.basename(checkpoint_path)
        # checkpoint_epoch_001_step_000500.pt
        parts = basename.split('_')
        for i, part in enumerate(parts):
            if part == 'epoch' and i + 1 < len(parts):
                return int(parts[i + 1])
        return 0

class MemoryOptimizedTrainer:
    """
    Trainer with memory optimization techniques
    """
    def __init__(self, model, use_mixed_precision=True):
        self.model = model
        self.use_mixed_precision = use_mixed_precision
        
        if use_mixed_precision:
            try:
                from torch.cuda.amp import GradScaler, autocast
                self.scaler = GradScaler()
                self.autocast = autocast
                print("✅ Mixed precision training enabled")
            except ImportError:
                print("⚠️ Mixed precision not available, using float32")
                self.use_mixed_precision = False
                self.scaler = None
                self.autocast = None
        else:
            self.scaler = None
            self.autocast = None
    
    def train_step(self, model, optimizer, batch, criterion, trg_pad, device, accumulation_steps=1):
        """
        Memory-optimized training step with gradient accumulation
        """
        model.train()
        
        src_img, trg = batch
        batch_size = src_img.size(0)
        
        # Split batch into micro-batches for gradient accumulation
        micro_batch_size = batch_size // accumulation_steps
        total_loss = 0
        
        optimizer.zero_grad()
        
        for i in range(accumulation_steps):
            start_idx = i * micro_batch_size
            end_idx = (i + 1) * micro_batch_size if i < accumulation_steps - 1 else batch_size
            
            micro_src = src_img[start_idx:end_idx]
            micro_trg = trg[start_idx:end_idx]
            
            # Prepare input and target
            trg_input = micro_trg[:, :-1]
            trg_target = micro_trg[:, 1:]
            
            # Create mask
            from utils.mask import create_mask
            trg_mask = create_mask(trg_input, trg_pad, True, device)
            
            if self.use_mixed_precision and self.autocast:
                with self.autocast():
                    preds = model(micro_src, trg_input, trg_mask)
                    preds = preds.contiguous().view(-1, preds.size(-1))
                    targets = trg_target.contiguous().view(-1)
                    loss = criterion(preds, targets) / accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                preds = model(micro_src, trg_input, trg_mask)
                preds = preds.contiguous().view(-1, preds.size(-1))
                targets = trg_target.contiguous().view(-1)
                loss = criterion(preds, targets) / accumulation_steps
                loss.backward()
            
            total_loss += loss.item()
            
            # Clear micro-batch from memory
            del micro_src, micro_trg, preds, targets, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Gradient clipping
        if self.use_mixed_precision and self.scaler:
            self.scaler.unscale_(optimizer.optimizer)  # optimizer.optimizer for scheduler wrapper
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step_and_update_lr()
        
        return total_loss
