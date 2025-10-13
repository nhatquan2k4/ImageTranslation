import torch
import torch.nn as nn
import json
import os
import sys
from argparse import ArgumentParser
from utils.reader import create_data
from utils.tokenizer import tokenizer
from utils.image_processor import ImageProcessor
from utils.checkpoint_manager import CheckpointManager
from modules.transformer import Transformer
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Windows multiprocessing fix
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

class SimpleDataset(Dataset):
    """Simple dataset without multiprocessing issues"""
    
    def __init__(self, df, tokenizer, vocab, max_length, image_processor):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Process image
            image_path = row['image_path']
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_processor.process_image(image, is_training=True)
            
            # Ensure image tensor has correct shape
            if image_tensor.shape[0] != 3:
                image_tensor = image_tensor[:3] if image_tensor.shape[0] > 3 else torch.cat([image_tensor, torch.zeros(3-image_tensor.shape[0], *image_tensor.shape[1:])])
            
            # Process text
            text = str(row['target_text']).strip()
            if not text:
                text = "empty"
                
            tokens = self.tokenizer.tokenize(text)
            tokens = ['<sos>'] + tokens + ['<eos>']
            
            # Convert to indices with bounds checking
            indices = []
            for token in tokens[:self.max_length-1]:  # Leave space for padding
                if token in self.vocab:
                    indices.append(self.vocab[token])
                else:
                    indices.append(self.vocab['<unk>'])
            
            # Ensure minimum length of 2
            if len(indices) < 2:
                indices = [self.vocab['<sos>'], self.vocab['<eos>']]
            
            # Pad to exactly max_length
            while len(indices) < self.max_length:
                indices.append(self.vocab['<pad>'])
            
            # Truncate if too long
            indices = indices[:self.max_length]
            
            return image_tensor, torch.tensor(indices, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return safe dummy data with guaranteed dimensions
            image_tensor = torch.zeros(3, self.image_processor.image_size, self.image_processor.image_size)
            indices = [self.vocab['<sos>'], self.vocab['<eos>']] + [self.vocab['<pad>']] * (self.max_length - 2)
            return image_tensor, torch.tensor(indices[:self.max_length], dtype=torch.long)

def create_simple_vocab(tokenizer, df_train):
    """Create vocabulary without torchtext"""
    print("Creating vocabulary...")
    
    vocab_words = set()
    for _, row in df_train.iterrows():
        tokens = tokenizer.tokenize(row["target_text"])
        vocab_words.update(tokens)
    
    # Create vocab dict
    specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
    vocab = {}
    
    for i, token in enumerate(specials):
        vocab[token] = i
    
    for i, word in enumerate(sorted(vocab_words), start=len(specials)):
        if word not in vocab:
            vocab[word] = i
    
    print(f"✅ Vocab created: {len(vocab)} tokens")
    return vocab

def simple_collate_fn(batch):
    """Simple collate function"""
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets = torch.stack(targets)
    return images, targets

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train one epoch with better error handling"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    print(f"\n📚 Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        try:
            # Validate tensor shapes
            if images.dim() != 4:
                print(f"❌ Invalid image tensor shape: {images.shape}")
                continue
                
            if targets.dim() != 2:
                print(f"❌ Invalid target tensor shape: {targets.shape}")
                continue
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Skip if sequence too short
            if targets.size(1) < 2:
                print(f"❌ Sequence too short: {targets.size(1)}")
                continue
            
            # Create mask - ensure proper dimensions
            tgt_input = targets[:, :-1]  # Remove last token for input
            tgt_len = tgt_input.size(1)
            
            if tgt_len <= 0:
                print(f"❌ Invalid target length: {tgt_len}")
                continue
                
            # Create causal mask
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device))
            
            # Ensure mask is boolean
            tgt_mask = (tgt_mask == 0)  # True where we want to mask
            
            # Forward pass with proper input shapes
            try:
                outputs = model(images, tgt_input, tgt_mask)
                
                if outputs.size(-1) == 0:
                    print("❌ Model output has 0 vocab size")
                    continue
                    
            except RuntimeError as e:
                print(f"❌ Model forward error: {e}")
                continue
            
            # Calculate loss
            try:
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets[:, 1:].reshape(-1)
                )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ Invalid loss: {loss}")
                    continue
                    
            except RuntimeError as e:
                print(f"❌ Loss calculation error: {e}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 5 == 0:  # Print more frequently
                avg_loss = total_loss / num_batches
                print(f"   Batch {batch_idx + 1}: Loss = {loss.item():.4f}, Avg = {avg_loss:.4f}")
                
        except Exception as e:
            print(f"❌ Unexpected error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)

def main(args):
    print("🚀 WINDOWS-SAFE TRAINING")
    print("=" * 50)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        
    # Load data
    print("📂 Loading data...")
    train_json = os.path.join(args.data_folder, 'train', 'train.json')
    df_train = create_data(train_json)
    
    # Take smaller subset for testing
    if len(df_train) > 1000:
        df_train = df_train.head(1000)
        print(f"📊 Using subset: {len(df_train)} samples")
    else:
        print(f"📊 Train samples: {len(df_train)}")
    
    # Create tokenizer and vocab
    print("📝 Creating tokenizer...")
    trg_tokenizer = tokenizer('vi')
    vocab = create_simple_vocab(trg_tokenizer, df_train)
    
    # Create image processor
    image_size = config.get('image_size', 224)
    image_processor = ImageProcessor(image_size=image_size)
    
    # Create dataset
    print("📚 Creating dataset...")
    max_length = config.get('max_strlen', config.get('max_seq_length', 64))
    
    dataset = SimpleDataset(df_train, trg_tokenizer, vocab, max_length, image_processor)
    
    # Create dataloader - NO multiprocessing
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('micro_batch_size', 2),
        shuffle=True,
        collate_fn=simple_collate_fn,
        num_workers=0,  # IMPORTANT: 0 for Windows
        pin_memory=False
    )
    
    print(f"✅ DataLoader created: {len(dataloader)} batches")
    
    # Create model
    print("🧠 Creating model...")
    model = Transformer(
        len(vocab),
        d_model=config.get('d_model', 256),  # Smaller for testing
        n=config.get('n_layers', 3),         # Fewer layers
        heads=config.get('heads', 4),        # Fewer heads
        dropout=config.get('dropout', 0.1),
        image_size=image_size
    ).to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Parameters: {total_params:,}")
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    # Create save folder
    os.makedirs(args.model_folder, exist_ok=True)
    
    # Training loop
    print("🏃‍♂️ Starting training...")
    
    n_epochs = config.get('epoch', 3)
    best_loss = float('inf')
    
    for epoch in range(1, n_epochs + 1):
        try:
            epoch_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch)
            print(f"📊 Epoch {epoch} - Avg Loss: {epoch_loss:.4f}")
            
            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(args.model_folder, 'best_model.pt'))
                torch.save(vocab, os.path.join(args.model_folder, 'vocab.pth'))
                print(f"✅ New best model saved! Loss: {best_loss:.4f}")
                
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted!")
            break
        except Exception as e:
            print(f"❌ Error in epoch {epoch}: {e}")
            continue
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.model_folder, 'final_model.pt'))
    print(f"🎉 Training completed! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    # Windows multiprocessing fix
    multiprocessing.freeze_support()
    
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/safe_config.json')
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--model_folder', type=str, required=True)
    
    args = parser.parse_args()
    main(args)