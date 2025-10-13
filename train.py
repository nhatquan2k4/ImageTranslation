import torch
import torch.nn as nn
import json
import os
from argparse import ArgumentParser
from utils.reader import create_data
from utils.tokenizer import tokenizer
from utils.dataset import create_dataset
from utils.scheduler import MyScheduler
from utils.loss import LabelSmoothingLoss
from utils.step import step
from utils.validation import validiate
from utils.image_processor import ImageProcessor
from utils.checkpoint_manager import CheckpointManager, MemoryOptimizedTrainer
from modules.transformer import Transformer
from torchtext.vocab import build_vocab_from_iterator


def yield_tokens(data_iter, tokenizer):
    for _, row in data_iter.iterrows():
        yield tokenizer.tokenize(row["target_text"])



def main(args):
    data_folder = args.data_folder
    model_folder = args.model_folder
    resume_from = args.resume_from_checkpoint

    print("Load config")
    config_file = open('config/config.json')
    cfg = json.load(config_file)
    d_model = cfg.get('d_model', 512)
    batch_size = cfg['batch_size']
    micro_batch_size = cfg.get('micro_batch_size', batch_size)
    gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', 1)
    max_strlen = cfg['max_strlen']
    n_epoch = cfg['epoch']
    print_every = cfg['print_every']
    save_every = cfg.get('save_every', 5)
    checkpoint_every_n_steps = cfg.get('checkpoint_every_n_steps', 500)
    n_layers = cfg['n_layers']
    heads = cfg.get('heads', 8)
    dropout = cfg['dropout']
    image_size = cfg.get('image_size', 224)
    mixed_precision = cfg.get('mixed_precision', True)
    max_checkpoints = cfg.get('max_checkpoints_to_keep', 3)
    print(json.dumps(cfg, indent=3))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Use {device}")
    
    # Create model folder
    os.makedirs(model_folder, exist_ok=True)

    print("Loading data")
    train_json = os.path.join(data_folder, 'train', 'train.json')
    val_json = os.path.join(data_folder, 'val', 'val.json')
    df_train = create_data(train_json)
    df_val = create_data(val_json)
    print(f"Train size: {len(df_train)}")
    print(f"Validate size: {len(df_val)}")

    print("Creating tokenizer")
    trg_tokenizer = tokenizer('vi')

    print("Building vocab")
    trg_vocab = build_vocab_from_iterator(
        yield_tokens(df_train, trg_tokenizer),
        specials=["<unk>", "<pad>", "<sos>", "<eos>"]
    )
    trg_vocab.set_default_index(trg_vocab["<unk>"])

    print("Creating image processor")
    image_processor = ImageProcessor(image_size=image_size)

    print("Construct dataset")
    # Use micro_batch_size for actual DataLoader
    train_dataset = create_dataset(df_train, trg_tokenizer, trg_vocab, micro_batch_size, max_strlen, device, image_processor)
    val_dataset = create_dataset(df_val, trg_tokenizer, trg_vocab, micro_batch_size, max_strlen, device, image_processor, istrain=False)

    print("Creating model")
    model = Transformer(
        len(trg_vocab),
        d_model=d_model,
        n=n_layers,
        heads=heads,
        dropout=dropout,
        image_size=image_size,
        vision_hidden_dim=cfg.get('vision_hidden_dim', 768),
        vision_layers=cfg.get('vision_layers', 6),
        vision_heads=cfg.get('vision_heads', 8),
        patch_size=cfg.get('patch_size', 16)
    ).to(device)

    print("Initializing weights")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print("Creating optimizer")
    opt = MyScheduler(
        torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.98),
            eps=1e-9
        ),
        init_lr=0.2,
        d_model=d_model,
        n_warmup=cfg.get('warmup_steps', 4000)
    )

    print("Creating loss function")
    criterion = LabelSmoothingLoss(
        len(trg_vocab),
        trg_pad_idx=trg_vocab["<pad>"],
        smoothing=cfg.get('label_smoothing', 0.1)
    )

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(model_folder, max_checkpoints)
    
    # Initialize memory optimized trainer
    memory_trainer = MemoryOptimizedTrainer(model, mixed_precision)

    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    best_loss = None
    
    if resume_from:
        if resume_from == "latest":
            checkpoint_path = checkpoint_manager.find_latest_checkpoint()
        else:
            checkpoint_path = resume_from
            
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch, start_step, last_loss = checkpoint_manager.load_checkpoint(
                checkpoint_path, model, opt
            )
            best_loss = last_loss
            print(f"📂 Resumed training from epoch {start_epoch}, step {start_step}")
        else:
            print(f"⚠️ Checkpoint not found: {resume_from}, starting from scratch")

    print("Training begin")
    global_step = start_step
    
    for epoch in range(start_epoch, n_epoch):
        total_loss = 0
        n_iter = 0
        
        for i, batch in enumerate(train_dataset):
            # Memory optimized training step
            loss = memory_trainer.train_step(
                model, opt, batch, criterion, 
                trg_vocab["<pad>"], device, 
                gradient_accumulation_steps
            )
            
            n_iter += 1
            total_loss += loss
            global_step += 1
            
            # Print progress
            if (i + 1) % print_every == 0:
                avg_loss = total_loss / n_iter
                print(f"epoch: {epoch+1:03d} - step: {global_step:06d} - iter: {i+1:04d} - loss: {avg_loss:.4f}")
                total_loss = 0
                n_iter = 0
            
            # Save checkpoint periodically
            if global_step % checkpoint_every_n_steps == 0:
                checkpoint_manager.save_checkpoint(
                    model, opt, epoch, global_step, loss, trg_vocab, cfg
                )
                # Clear cache after checkpoint
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Validation at end of epoch
        print(f"🔍 Running validation for epoch {epoch+1}...")
        valid_loss = validiate(model, val_dataset, criterion, trg_vocab["<pad>"], device)
        print(f'epoch: {epoch+1:03d} - valid loss: {valid_loss:.4f}')
        
        # Save epoch checkpoint
        torch.save(model.state_dict(), os.path.join(model_folder, f'model.{epoch+1}.pt'))
        
        # Save best model
        if best_loss is None or best_loss > valid_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(model_folder, 'model_best.pt'))
            torch.save(trg_vocab, os.path.join(model_folder, 'trg_vocab.pth'))
            # Copy config to model folder for inference
            import shutil
            shutil.copy('config/config.json', os.path.join(model_folder, 'config.json'))
            print(f'✅ Best model saved with validation loss: {best_loss:.4f}')
        
        # Save checkpoint every few epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_manager.save_checkpoint(
                model, opt, epoch + 1, global_step, valid_loss, trg_vocab, cfg
            )
    
    print("🎉 Training completed!")
    print(f"📊 Final best validation loss: {best_loss:.4f}")
    print(f"💾 Model saved to: {model_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_folder', type=str, required=True, help="Path to data folder")
    parser.add_argument('-m', '--model_folder', type=str, required=True, help="Path to model folder")
    parser.add_argument('-r', '--resume_from_checkpoint', type=str, default=None, 
                       help="Path to checkpoint file or 'latest' to resume from latest checkpoint")
    args = parser.parse_args()
    main(args)
