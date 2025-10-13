import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class ImageTranslationDataset(Dataset):
    def __init__(self, df, tokenizer, vocab, max_strlen, image_processor):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_strlen = max_strlen
        self.image_processor = image_processor
        
        # Filter out sequences that are too long
        self.df = self.df[self.df['target_text'].str.split().str.len() <= max_strlen - 2]  # -2 for <sos> and <eos>
        
        print(f"Dataset created with {len(self.df)} samples after filtering")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load và process image
        try:
            image = Image.open(row['image_path']).convert('RGB')
            image_tensor = self.image_processor.process_image(image)
        except Exception as e:
            print(f"Error loading image {row['image_path']}: {e}")
            # Return a dummy image tensor if loading fails
            image_tensor = torch.zeros(3, self.image_processor.image_size, self.image_processor.image_size)
        
        # Tokenize Vietnamese target text
        target_tokens = self.tokenizer.tokenize(row['target_text'])
        
        # Add special tokens
        target_tokens = ['<sos>'] + target_tokens + ['<eos>']
        
        # Convert to indices, handle unknown tokens
        target_indices = []
        for token in target_tokens:
            if token in self.vocab:
                target_indices.append(self.vocab[token])
            else:
                target_indices.append(self.vocab['<unk>'])
        
        return {
            'image': image_tensor,
            'target': torch.tensor(target_indices, dtype=torch.long),
            'target_length': len(target_indices)
        }


def collate_fn(batch, pad_idx, max_length):
    """
    Collate function để batch các samples
    """
    images = torch.stack([item['image'] for item in batch])
    target_lengths = [item['target_length'] for item in batch]
    
    # Pad target sequences to max length in batch
    max_len = min(max(target_lengths), max_length)
    padded_targets = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    
    for i, item in enumerate(batch):
        target = item['target']
        seq_len = min(len(target), max_len)
        padded_targets[i, :seq_len] = target[:seq_len]
    
    return images, padded_targets


def create_dataset(df, tokenizer, vocab, batch_size, max_strlen, device, image_processor, istrain=True):
    """
    Tạo DataLoader cho image-to-translation task
    """
    dataset = ImageTranslationDataset(df, tokenizer, vocab, max_strlen, image_processor)
    
    def collate_wrapper(batch):
        images, targets = collate_fn(batch, vocab["<pad>"], max_strlen)
        return images.to(device), targets.to(device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=istrain,
        collate_fn=collate_wrapper,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return dataloader
