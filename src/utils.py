# utils.py
import json
import torch
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)




def save_checkpoint(path, model, optimizer, epoch, args=None):
    ck = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'args': args
    }
    torch.save(ck, path)




def load_checkpoint(path, model, optimizer=None, map_location=None):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ck:
        optimizer.load_state_dict(ck['optimizer_state_dict'])
    return ck