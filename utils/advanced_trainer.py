import math
from collections import defaultdict
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


class AdvancedTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device | None = None,
        *,
        criterion: nn.Module | None = None,
        gradient_accumulation_steps: int = 4,
        mixed_precision: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=0)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self.max_grad_norm = float(max_grad_norm)

        # Fix deprecated GradScaler warning
        if self.mixed_precision:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        self.metrics = defaultdict(list)
        self.best_loss = float("inf")

        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def _create_mask(self, trg: torch.Tensor):
        """Tạo causal mask (B, T, T) từ trg (B, T)."""
        if trg.dim() != 2:
            return None
        B, T = trg.shape
        if T <= 0:
            return None
        mask = torch.tril(torch.ones(T, T, device=self.device, dtype=torch.bool))
        return mask.unsqueeze(0).expand(B, -1, -1)

    def _step_scheduler(self):
        if self.scheduler is None:
            return
        if hasattr(self.scheduler, "step_and_update_lr"):
            self.scheduler.step_and_update_lr()
        else:
            self.scheduler.step()

    def train_step(self, dataloader, epoch: int) -> float:
        """Train 1 epoch, trả về average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = max(1, len(dataloader))
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, targets) in enumerate(dataloader):
            try:
                # Kiểm tra dữ liệu
                if images.dim() != 4 or targets.dim() != 2:
                    print(f"❌ Invalid shapes: images={tuple(images.shape)}, targets={tuple(targets.shape)}")
                    continue
                if targets.size(1) < 2:
                    print(f"❌ Sequence too short: {targets.size(1)}")
                    continue

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                trg_input = targets[:, :-1]
                trg_output = targets[:, 1:]
                if trg_input.numel() == 0 or trg_output.numel() == 0:
                    continue

                trg_mask = self._create_mask(trg_input)
                if trg_mask is None:
                    continue

                # Forward & loss
                with autocast('cuda', enabled=self.mixed_precision):
                    outputs = self.model(images, trg_input, trg_mask)  # (B, T-1, V)
                    if outputs.dim() != 3 or outputs.size(-1) == 0:
                        print("❌ Model output shape invalid.")
                        continue

                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        trg_output.reshape(-1),
                    )
                    loss = loss / self.gradient_accumulation_steps  # accumulation

                # Backward
                if self.scaler and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item()

                # Cập nhật mỗi accumulation step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler and self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    if self.scaler and self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self._step_scheduler()

                # Log nhẹ
                if (batch_idx + 1) % 50 == 0:
                    curr_lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch} | Batch {batch_idx+1}/{num_batches} | "
                        f"Loss: {(loss.item()*self.gradient_accumulation_steps):.4f} | LR: {curr_lr:.2e}"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"❌ CUDA OOM at batch {batch_idx}, skipping…")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                print(f"❌ RuntimeError at batch {batch_idx}: {e}")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            except Exception as e:
                print(f"❌ Unexpected error at batch {batch_idx}: {e}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

        # Nếu còn gradient (batch không chia hết)
        if (len(dataloader) % self.gradient_accumulation_steps) != 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._step_scheduler()

        avg_loss = total_loss / num_batches
        self.metrics["train_loss"].append(avg_loss)
        self.best_loss = min(self.best_loss, avg_loss)
        return avg_loss

    def save_checkpoint(self, filepath: str, epoch: int, loss: float, vocab=None, config: dict | None = None):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self.scheduler, "state_dict") else None,
            "loss": loss,
            "best_loss": self.best_loss,
            "metrics": dict(self.metrics),
        }
        if vocab is not None:
            ckpt["vocab"] = vocab
        if config is not None:
            ckpt["config"] = config
        if self.scaler.is_enabled():
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(ckpt, filepath)
        print(f"✅ Checkpoint saved to: {filepath}")

    def get_memory_usage(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_GB": torch.cuda.memory_allocated() / (1024 ** 3),
            "reserved_GB": torch.cuda.memory_reserved() / (1024 ** 3),
            "max_allocated_GB": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }


class ImprovedScheduler:
    """
    Warmup (công thức Transformer) + Cosine annealing.
    Gọi step_and_update_lr() mỗi lần cập nhật trọng số.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, max_steps: int = 100000):
        self.optimizer = optimizer
        self.d_model = int(d_model)
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_steps = max(self.warmup_steps + 1, int(max_steps))
        self.step_num = 0

    def step_and_update_lr(self):
        self.step_num += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self) -> float:
        # Warmup
        if self.step_num <= self.warmup_steps:
            return self.step_num * (self.d_model ** -0.5) * (self.warmup_steps ** -1.5)

        # Cosine về 0
        progress = (self.step_num - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_lr = (self.d_model ** -0.5) * (self.warmup_steps ** -0.5)
        return base_lr * cosine_decay

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state_dict: dict):
        self.step_num = int(state_dict.get("step_num", 0))
        self.d_model = int(state_dict.get("d_model", self.d_model))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.max_steps = int(state_dict.get("max_steps", self.max_steps))
