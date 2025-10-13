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
from utils.advanced_trainer import AdvancedTrainer, ImprovedScheduler
from utils.validation_metrics import ValidationMetrics, EarlyStopping
from modules.transformer import Transformer


def yield_tokens(data_iter, tokenizer):
    for _, row in data_iter.iterrows():
        yield tokenizer.tokenize(row["target_text"])



def main(args):
    data_folder = args.data_folder
    model_folder = args.model_folder
    resume_from = args.resume_from_checkpoint
    config_path = args.config

    print(f"Load config from: {config_path}")
    config_file = open(config_path)
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
    # Custom vocab builder to avoid torchtext dependency issues
    vocab_words = set()
    for _, row in df_train.iterrows():
        tokens = trg_tokenizer.tokenize(row["target_text"])
        vocab_words.update(tokens)
    
    # Create vocab dictionary
    specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
    trg_vocab = {}
    
    # Add special tokens first
    for i, token in enumerate(specials):
        trg_vocab[token] = i
    
    # Add regular tokens
    for i, word in enumerate(sorted(vocab_words), start=len(specials)):
        if word not in trg_vocab:
            trg_vocab[word] = i
    
    print(f"Vocabulary size: {len(trg_vocab)}")

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

    # Initialize enhanced training components
    checkpoint_manager = CheckpointManager(model_folder, max_checkpoints)
    
    # Use improved scheduler
    improved_scheduler = ImprovedScheduler(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
        d_model=d_model,
        warmup_steps=cfg.get('warmup_steps', 4000),
        max_steps=n_epoch * len(train_dataset)
    )
    
    # Initialize advanced trainer with optimizations
    trainer = AdvancedTrainer(
        model=model,
        optimizer=improved_scheduler.optimizer,
        scheduler=improved_scheduler,
        criterion=criterion,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        max_grad_norm=cfg.get('max_grad_norm', 1.0)
    )
    
    # Initialize validation metrics and early stopping
    val_metrics = ValidationMetrics(trg_vocab, trg_tokenizer, device)
    early_stopping = EarlyStopping(
        patience=cfg.get('early_stopping_patience', 5),
        min_delta=0.001,
        mode='min'
    )

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
                checkpoint_path, model, improved_scheduler.optimizer
            )
            best_loss = last_loss
            print(f"📂 Resumed training from epoch {start_epoch}, step {start_step}")
        else:
            print(f"⚠️ Checkpoint not found: {resume_from}, starting from scratch")

    print("🚀 Training begin with enhanced optimizations")
    global_step = start_step
    
    for epoch in range(start_epoch, n_epoch):
        print(f"\n📚 Epoch {epoch+1}/{n_epoch}")
        
        # Enhanced training step
        epoch_loss = trainer.train_step(train_dataset, epoch)
        global_step += len(train_dataset)
        
        # Comprehensive validation
        print(f"🔍 Running validation for epoch {epoch+1}...")
        validation_results = val_metrics.calculate_metrics(
            model, val_dataset, max_samples=500  # Limit for speed
        )
        
        val_loss = validation_results.get('loss', epoch_loss)  # Fallback to train loss
        
        print(f'📊 Epoch {epoch+1} Results:')
        print(f'   Train Loss: {epoch_loss:.4f}')
        print(f'   Val Accuracy: {validation_results["accuracy"]:.3f}')
        print(f'   Val CER: {validation_results["cer"]:.3f}')
        print(f'   Val WER: {validation_results["wer"]:.3f}')
        print(f'   Val BLEU: {validation_results["bleu"]:.3f}')
        
        # Memory usage monitoring
        if torch.cuda.is_available():
            memory_stats = trainer.get_memory_usage()
            print(f'   GPU Memory: {memory_stats["allocated"]:.1f}GB allocated, '
                  f'{memory_stats["reserved"]:.1f}GB reserved')
        
        # Early stopping check
        early_stop_result = early_stopping(validation_results["cer"])  # Use CER as metric
        
        if early_stop_result['improved']:
            print(f"✅ New best CER: {early_stop_result['best_score']:.4f}")
            # Save best model
            torch.save(model.state_dict(), os.path.join(model_folder, 'model_best.pt'))
            torch.save(trg_vocab, os.path.join(model_folder, 'trg_vocab.pth'))
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(
                checkpoint_manager, epoch, global_step, epoch_loss, trg_vocab, cfg
            )
        
        # Early stopping
        if early_stop_result['should_stop']:
            print(f"🛑 Early stopping triggered! No improvement for {early_stopping.patience} epochs")
            break
        

    
    print("🎉 Training completed!")
    print(f"📊 Best CER achieved: {early_stopping.best_score:.4f}")
    print(f"💾 Model saved to: {model_folder}")
    
    # Copy config to model folder for inference
    import shutil
    shutil.copy('config/config.json', os.path.join(model_folder, 'config.json'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_folder', type=str, required=True, help="Path to data folder")
    parser.add_argument('-m', '--model_folder', type=str, required=True, help="Path to model folder")
    parser.add_argument('-c', '--config', type=str, default='config/config.json', help="Path to config file")
    parser.add_argument('-r', '--resume_from_checkpoint', type=str, default=None, 
                       help="Path to checkpoint file or 'latest' to resume from latest checkpoint")
    args = parser.parse_args()
    main(args)
