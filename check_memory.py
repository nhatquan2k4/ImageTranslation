#!/usr/bin/env python3
"""
GPU Memory checker and config optimizer
"""

import torch
import json
import psutil
import os
from modules.transformer import Transformer

def check_system_resources():
    """Check system memory and GPU resources"""
    print("=" * 60)
    print("🖥️  SYSTEM RESOURCE CHECK")
    print("=" * 60)
    
    # CPU and RAM
    cpu_count = psutil.cpu_count()
    ram_total = psutil.virtual_memory().total / (1024**3)  # GB
    ram_available = psutil.virtual_memory().available / (1024**3)  # GB
    
    print(f"💾 CPU Cores: {cpu_count}")
    print(f"💾 RAM Total: {ram_total:.1f}GB")
    print(f"💾 RAM Available: {ram_available:.1f}GB")
    
    # GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"🎮 GPU {i}: {gpu_name}")
            print(f"🎮 GPU {i} Memory: {gpu_memory:.1f}GB")
            
            # Test current memory usage
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            free = gpu_memory - cached
            
            print(f"🎮 GPU {i} Used: {allocated:.1f}GB, Cached: {cached:.1f}GB, Free: {free:.1f}GB")
    else:
        print("⚠️  No CUDA GPU detected - will use CPU training (very slow)")
        
    return {
        'cpu_cores': cpu_count,
        'ram_total': ram_total,
        'ram_available': ram_available,
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    }

def estimate_model_memory(config):
    """Estimate memory usage of model with given config"""
    
    d_model = config.get('d_model', 512)
    n_layers = config.get('n_layers', 6)
    heads = config.get('heads', 8)
    vision_hidden_dim = config.get('vision_hidden_dim', 768)
    vision_layers = config.get('vision_layers', 6)
    batch_size = config.get('batch_size', 4)
    max_strlen = config.get('max_strlen', 128)
    image_size = config.get('image_size', 224)
    vocab_size = 10000  # Estimate
    
    # Rough estimation of parameter count (in millions)
    # Vision Transformer parameters
    patch_size = config.get('patch_size', 16)
    num_patches = (image_size // patch_size) ** 2
    vision_params = (
        vision_hidden_dim * 3 * patch_size * patch_size +  # Patch embedding
        vision_hidden_dim * num_patches +  # Position embedding
        vision_layers * (
            4 * vision_hidden_dim * vision_hidden_dim +  # Attention layers
            8 * vision_hidden_dim * vision_hidden_dim     # FFN layers
        ) +
        vision_hidden_dim * d_model  # Projection layer
    )
    
    # Text decoder parameters
    text_params = (
        vocab_size * d_model +  # Embedding
        d_model * max_strlen +  # Position embedding
        n_layers * (
            4 * d_model * d_model +  # Self-attention
            4 * d_model * d_model +  # Cross-attention
            8 * d_model * d_model    # FFN
        ) +
        d_model * vocab_size  # Output projection
    )
    
    total_params = (vision_params + text_params) / 1e6  # Convert to millions
    
    # Memory estimation (rough)
    # Parameters: 4 bytes per parameter (float32)
    # Gradients: 4 bytes per parameter
    # Optimizer states (Adam): 8 bytes per parameter (momentum + variance)
    # Activations: depends on batch size and sequence length
    
    param_memory = total_params * 4 / 1024  # MB
    grad_memory = total_params * 4 / 1024   # MB
    optimizer_memory = total_params * 8 / 1024  # MB
    
    # Activation memory (rough estimate)
    activation_memory = batch_size * (
        3 * image_size * image_size * 4 +  # Input image
        num_patches * vision_hidden_dim * 4 +  # Vision features
        max_strlen * d_model * 4 * n_layers  # Text activations
    ) / (1024 * 1024)  # Convert to MB
    
    total_memory_mb = param_memory + grad_memory + optimizer_memory + activation_memory
    total_memory_gb = total_memory_mb / 1024
    
    return {
        'parameters_millions': total_params,
        'memory_breakdown': {
            'parameters_mb': param_memory,
            'gradients_mb': grad_memory,
            'optimizer_mb': optimizer_memory,
            'activations_mb': activation_memory
        },
        'total_memory_gb': total_memory_gb
    }

def recommend_config(available_memory_gb):
    """Recommend optimal config based on available memory"""
    
    configs = {
        "low_memory": {  # For 2-4GB GPU
            "description": "Low Memory (2-4GB GPU)",
            "d_model": 256,
            "batch_size": 2,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "n_layers": 4,
            "heads": 4,
            "vision_layers": 4,
            "vision_heads": 4,
            "vision_hidden_dim": 512,
            "mixed_precision": True
        },
        "medium_memory": {  # For 4-8GB GPU
            "description": "Medium Memory (4-8GB GPU)",
            "d_model": 384,
            "batch_size": 4,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "n_layers": 6,
            "heads": 6,
            "vision_layers": 6,
            "vision_heads": 6,
            "vision_hidden_dim": 768,
            "mixed_precision": True
        },
        "high_memory": {  # For 8-12GB GPU
            "description": "High Memory (8-12GB GPU)",
            "d_model": 512,
            "batch_size": 6,
            "micro_batch_size": 3,
            "gradient_accumulation_steps": 2,
            "n_layers": 6,
            "heads": 8,
            "vision_layers": 6,
            "vision_heads": 8,
            "vision_hidden_dim": 768,
            "mixed_precision": True
        },
        "very_high_memory": {  # For 12GB+ GPU
            "description": "Very High Memory (12GB+ GPU)",
            "d_model": 512,
            "batch_size": 8,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "n_layers": 6,
            "heads": 8,
            "vision_layers": 6,
            "vision_heads": 8,
            "vision_hidden_dim": 768,
            "mixed_precision": True
        }
    }
    
    if available_memory_gb <= 4:
        return configs["low_memory"]
    elif available_memory_gb <= 8:
        return configs["medium_memory"]
    elif available_memory_gb <= 12:
        return configs["high_memory"]
    else:
        return configs["very_high_memory"]

def main():
    print("🔍 Checking system resources and optimizing config...")
    
    # Check system resources
    resources = check_system_resources()
    
    # Load current config
    config_path = "config/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            current_config = json.load(f)
        print(f"\n📋 Current config loaded from {config_path}")
    else:
        print(f"\n⚠️  Config file not found: {config_path}")
        return
    
    # Estimate memory usage
    print("\n" + "=" * 60)
    print("📊 MEMORY ESTIMATION")
    print("=" * 60)
    
    memory_estimate = estimate_model_memory(current_config)
    print(f"🔢 Estimated parameters: {memory_estimate['parameters_millions']:.1f}M")
    print(f"💾 Estimated memory usage: {memory_estimate['total_memory_gb']:.1f}GB")
    
    breakdown = memory_estimate['memory_breakdown']
    print(f"   - Parameters: {breakdown['parameters_mb']:.0f}MB")
    print(f"   - Gradients: {breakdown['gradients_mb']:.0f}MB") 
    print(f"   - Optimizer: {breakdown['optimizer_mb']:.0f}MB")
    print(f"   - Activations: {breakdown['activations_mb']:.0f}MB")
    
    # Check if current config fits in available memory
    available_memory = resources.get('gpu_memory', 0)
    estimated_usage = memory_estimate['total_memory_gb']
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if available_memory == 0:
        print("⚠️  No GPU detected - training will be very slow on CPU")
        print("💡 Consider using Google Colab, Kaggle, or cloud GPU services")
        return
    
    usage_ratio = estimated_usage / available_memory
    
    if usage_ratio > 0.9:
        print(f"❌ Current config requires {estimated_usage:.1f}GB but only {available_memory:.1f}GB available")
        print("🔧 Recommending optimized config...")
        
        # Get recommended config
        recommended = recommend_config(available_memory * 0.8)  # Use 80% of available memory
        
        print(f"\n✅ Recommended: {recommended['description']}")
        print("📝 Optimized parameters:")
        for key, value in recommended.items():
            if key != 'description':
                current_value = current_config.get(key, 'N/A')
                if current_value != value:
                    print(f"   {key}: {current_value} → {value}")
        
        # Ask if user wants to apply recommendations
        response = input("\n❓ Apply recommended config? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            # Update config
            current_config.update(recommended)
            # Remove description key
            current_config.pop('description', None)
            
            # Backup original
            backup_path = config_path + ".backup"
            if os.path.exists(config_path):
                os.rename(config_path, backup_path)
                print(f"💾 Original config backed up to: {backup_path}")
            
            # Save new config
            with open(config_path, 'w') as f:
                json.dump(current_config, f, indent=2)
            print(f"✅ Optimized config saved to: {config_path}")
        else:
            print("⏭️  Keeping current config (may cause out-of-memory errors)")
            
    elif usage_ratio > 0.7:
        print(f"⚠️  Current config uses {usage_ratio*100:.0f}% of available memory")
        print("💡 Consider reducing batch_size if you encounter memory issues")
        
    else:
        print(f"✅ Current config looks good ({usage_ratio*100:.0f}% memory usage)")
        print("💡 You might be able to increase batch_size for faster training")
    
    print("\n🚀 Use these commands to start training:")
    print(f"   Normal training: python train.py -d /path/to/data -m /path/to/model")
    print(f"   Incremental training: python incremental_train.py -d /path/to/data -m /path/to/model")

if __name__ == "__main__":
    main()
