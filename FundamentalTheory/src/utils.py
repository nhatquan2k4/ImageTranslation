# utils.py
import json
import torch
from pathlib import Path
from typing import Optional, Any


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    args: Optional[Any] = None
):
    """Save model checkpoint with loss."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ck = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'args': vars(args) if args else None  # Serialize args as dict
    }
    torch.save(ck, path)
    print(f"✅ Checkpoint saved: {path} (Epoch {epoch}, Loss: {loss:.4f})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None
) -> dict:
    """Load model checkpoint."""
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ck:
        optimizer.load_state_dict(ck['optimizer_state_dict'])
    print(f"✅ Checkpoint loaded: {path} (Epoch: {ck.get('epoch', 'N/A')}, Loss: {ck.get('loss', 'N/A'):.4f})")
    return ck