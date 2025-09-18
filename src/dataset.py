import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from typing import List, Tuple


class ImgToTextDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=128, img_size=224, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = data
        self.tokenizer = tokenizer  # sentencepiece processor hoặc Huggingface tokenizer
        self.max_len = max_len
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = Path(s['image_path'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        tgt_text = s['target_text']

        # Encode text
        enc = self.tokenizer.encode(tgt_text)
        if hasattr(enc, "ids"):  # HuggingFace tokenizers
            ids = enc.ids
        else:  # sentencepiece -> list[int]
            ids = enc if isinstance(enc, list) else self.tokenizer.encode(tgt_text, out_type=int)

        # Nếu rỗng thì thay = [BOS, EOS]
        if len(ids) == 0:
            print(f"[WARNING] Empty target_text at index {idx}, raw text: '{tgt_text}'")
            ids = [1, 2]

        # Thêm BOS=1, EOS=2
        if ids[0] != 1:
            ids = [1] + ids
        if ids[-1] != 2:
            ids = ids + [2]

        # Pad / cut
        if len(ids) < self.max_len:
            ids = ids + [3] * (self.max_len - len(ids))  # PAD=3
        else:
            ids = ids[:self.max_len]
            if ids[-1] != 2:
                ids[-1] = 2

        tgt = torch.tensor(ids, dtype=torch.long)
        return img, tgt


# collate function cho DataLoader
def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    tgts = torch.stack([b[1] for b in batch], dim=0)
    return imgs, tgts
