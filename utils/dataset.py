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
        try:
            row = self.df.iloc[idx]
            
            # Load và process image
            try:
                image = Image.open(row['image_path']).convert('RGB')
                image_tensor = self.image_processor.process_image(image, is_training=True)
                
                # Ensure image has correct shape
                if image_tensor.shape != (3, self.image_processor.image_size, self.image_processor.image_size):
                    image_tensor = torch.zeros(3, self.image_processor.image_size, self.image_processor.image_size)
                    
            except Exception as e:
                print(f"Error loading image {row['image_path']}: {e}")
                image_tensor = torch.zeros(3, self.image_processor.image_size, self.image_processor.image_size)
            
            # Tokenize Vietnamese target text
            try:
                target_tokens = self.tokenizer.tokenize(row['target_text'])
            except:
                target_tokens = []
            
            # Add special tokens and limit length
            target_tokens = ['<sos>'] + target_tokens[:self.max_strlen-2] + ['<eos>']
            
            # Convert to indices
            target_indices = []
            for token in target_tokens:
                if token in self.vocab:
                    target_indices.append(self.vocab[token])
                else:
                    target_indices.append(self.vocab['<unk>'])
            
            # Ensure minimum length
            if len(target_indices) < 2:
                target_indices = [self.vocab['<sos>'], self.vocab['<eos>']]
            
            return {
                'image': image_tensor,
                'target': torch.tensor(target_indices, dtype=torch.long),
                'target_length': len(target_indices)
            }
            
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            # Return safe fallback
            image_tensor = torch.zeros(3, self.image_processor.image_size, self.image_processor.image_size)
            target_indices = [self.vocab['<sos>'], self.vocab['<eos>']]
            
            return {
                'image': image_tensor,
                'target': torch.tensor(target_indices, dtype=torch.long),
                'target_length': len(target_indices)
            }


def collate_fn(batch, pad_idx, max_length):
    """
    Collate function để batch các samples - Fixed tensor size issues
    """
    # Check if all images have same size
    images = []
    for item in batch:
        img = item['image']
        if img.dim() != 3:
            print(f"Warning: Image has {img.dim()} dimensions, expected 3")
            img = img.squeeze() if img.dim() > 3 else img.unsqueeze(0)
        images.append(img)
    
    try:
        images = torch.stack(images)
    except RuntimeError as e:
        print(f"Error stacking images: {e}")
        # Fallback: resize all to same size
        target_size = images[0].shape
        fixed_images = []
        for img in images:
            if img.shape != target_size:
                img = torch.zeros_like(images[0])
            fixed_images.append(img)
        images = torch.stack(fixed_images)
    
    # Fixed target padding
    target_lengths = [item['target_length'] for item in batch]
    max_len = max_length  # Use fixed max length instead of dynamic
    
    padded_targets = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    
    for i, item in enumerate(batch):
        target = item['target']
        seq_len = min(len(target), max_len)
        if seq_len > 0:
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
        num_workers=0,  # Set to 0 for Windows compatibility - fixes pickle error
        pin_memory=False,  # Disable pin_memory to avoid multiprocessing issues
    )
    
    return dataloader
