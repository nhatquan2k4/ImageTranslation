"""
Configuration file for Image-to-Text Translation model
"""
import torch


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    """Configuration class for model training and inference"""
    
    # ==================== Paths ====================
    data_root = "./data"
    json_file = "train/dataset.json"
    checkpoint_path = "./checkpoints/resnet_safe_gradual_best.pth"
    output_dir = "./outputs"
    
    # ==================== Architecture ====================
    encoder_type = "resnet101"
    encoder_dim = 2048
    embed_dim = 512
    n_heads = 8
    n_layers = 3
    ffn_dim = 2048
    dropout = 0.1
    
    # ==================== Learning Rates ====================
    encoder_lr = 2e-6
    decoder_lr = 2e-5
    weight_decay = 0.01
    
    # ==================== Training ====================
    batch_size = 12
    num_workers = 4
    epochs = 40
    start_epoch = 0
    max_samples = 60000
    gradient_accumulation_steps = 4
    
    # ==================== Training Phases ====================
    phase1_epochs = 15      # Frozen encoder phase
    phase2_epochs = 26      # Moderate encoder LR
    phase3_epochs = 40      # Normal dual LR
    
    # ==================== Image Processing ====================
    target_h = 224
    min_w = 224
    max_w = 1890
    max_w_batch = 1280
    
    # ==================== Inference ====================
    beam_size = 5
    max_length = 70
    repetition_penalty = 1.2
    length_penalty = 0.6
    
    # ==================== Optimization ====================
    use_amp = True
    label_smoothing = 0.0
    clip_grad_norm = 1.0
    early_stopping_patience = 3
    eval_samples = 2400
    warmup_epochs = 3
    
    # ==================== Data Split ====================
    train_ratio = 0.7
    random_seed = 42
    
    # ==================== Tokenizer ====================
    tokenizer_name = "vinai/bartpho-word"
    
    def __repr__(self):
        """Pretty print configuration"""
        config_str = "\n" + "="*70 + "\n"
        config_str += "MODEL CONFIGURATION\n"
        config_str += "="*70 + "\n"
        config_str += f"Device: {device}\n"
        config_str += f"Encoder: {self.encoder_type} (dim={self.encoder_dim})\n"
        config_str += f"Embedding dim: {self.embed_dim}\n"
        config_str += f"Transformer: {self.n_layers} layers, {self.n_heads} heads\n"
        config_str += f"Batch size: {self.batch_size}\n"
        config_str += f"Learning rates: encoder={self.encoder_lr:.1e}, decoder={self.decoder_lr:.1e}\n"
        config_str += "="*70
        return config_str


# Create default config instance
CONFIG = Config()
