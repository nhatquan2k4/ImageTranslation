#!/usr/bin/env python3
"""
Incremental training script for limited GPU memory
"""

import os
import json
import argparse
import subprocess
import time
from datetime import datetime

def get_gpu_memory():
    """Get GPU memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
            cached_memory = torch.cuda.memory_reserved(0) / 1024**3  # GB
            return {
                'total': total_memory,
                'allocated': allocated_memory, 
                'cached': cached_memory,
                'free': total_memory - cached_memory
            }
    except Exception as e:
        print(f"Could not get GPU memory: {e}")
    return None

def adjust_config_for_memory(config_path, target_memory_gb=4):
    """
    Adjust config based on available GPU memory
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Memory optimization settings based on target memory
    if target_memory_gb <= 4:  # Low memory (RTX 3060, GTX 1060, etc.)
        config.update({
            "batch_size": 2,
            "micro_batch_size": 1, 
            "gradient_accumulation_steps": 4,
            "d_model": 256,
            "n_layers": 4,
            "heads": 4,
            "vision_layers": 4,
            "vision_heads": 4,
            "mixed_precision": True
        })
        print("🔧 Low memory config applied (4GB or less)")
        
    elif target_memory_gb <= 8:  # Medium memory (RTX 3070, RTX 2080, etc.)
        config.update({
            "batch_size": 4,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2, 
            "d_model": 384,
            "n_layers": 6,
            "heads": 6,
            "vision_layers": 6,
            "vision_heads": 6,
            "mixed_precision": True
        })
        print("🔧 Medium memory config applied (4-8GB)")
        
    elif target_memory_gb <= 12:  # High memory (RTX 3080, RTX 4070, etc.)
        config.update({
            "batch_size": 6,
            "micro_batch_size": 3,
            "gradient_accumulation_steps": 2,
            "d_model": 512,
            "n_layers": 6,
            "heads": 8,
            "vision_layers": 6,
            "vision_heads": 8,
            "mixed_precision": True
        })
        print("🔧 High memory config applied (8-12GB)")
        
    else:  # Very high memory (RTX 4080, RTX 4090, etc.)
        config.update({
            "batch_size": 8,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "d_model": 512,
            "n_layers": 6,
            "heads": 8,
            "vision_layers": 6,
            "vision_heads": 8,
            "mixed_precision": True
        })
        print("🔧 Very high memory config applied (12GB+)")
    
    # Save adjusted config
    backup_path = config_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(config_path, backup_path)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Original config backed up to: {backup_path}")
    print(f"✅ Updated config saved to: {config_path}")
    
    return config

def incremental_train(data_folder, model_folder, max_epochs_per_session=5, target_memory_gb=None):
    """
    Train model incrementally with memory management
    """
    config_path = "config/config.json"
    
    # Auto-detect memory if not specified
    if target_memory_gb is None:
        gpu_info = get_gpu_memory()
        if gpu_info:
            target_memory_gb = gpu_info['total'] * 0.8  # Use 80% of available memory
            print(f"🔍 Detected GPU memory: {gpu_info['total']:.1f}GB, using {target_memory_gb:.1f}GB")
        else:
            target_memory_gb = 4  # Conservative default
            print("⚠️ Could not detect GPU memory, using conservative 4GB setting")
    
    # Adjust config for memory
    config = adjust_config_for_memory(config_path, target_memory_gb)
    total_epochs = config.get('epoch', 50)
    
    print(f"\n🚀 Starting incremental training")
    print(f"📊 Total epochs: {total_epochs}")
    print(f"📦 Epochs per session: {max_epochs_per_session}")
    print(f"💻 Target memory usage: {target_memory_gb:.1f}GB")
    
    current_epoch = 0
    session = 1
    
    while current_epoch < total_epochs:
        # Calculate epochs for this session
        epochs_this_session = min(max_epochs_per_session, total_epochs - current_epoch)
        
        print(f"\n" + "="*60)
        print(f"📅 Session {session}: Training epochs {current_epoch + 1}-{current_epoch + epochs_this_session}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Update config for this session
        config['epoch'] = epochs_this_session
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Prepare training command
        cmd = [
            "python", "train.py",
            "-d", data_folder,
            "-m", model_folder
        ]
        
        # Add resume flag if not first session
        if current_epoch > 0:
            cmd.extend(["-r", "latest"])
        
        # Run training
        try:
            start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=False)
            end_time = time.time()
            
            session_time = end_time - start_time
            print(f"\n✅ Session {session} completed in {session_time/60:.1f} minutes")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Session {session} failed with error: {e}")
            print("🔧 Try reducing batch_size or d_model in config and restart")
            return False
        except KeyboardInterrupt:
            print(f"\n⏹️ Training interrupted by user during session {session}")
            print("💡 You can resume later with: python train.py -d {data_folder} -m {model_folder} -r latest")
            return False
        
        current_epoch += epochs_this_session
        session += 1
        
        # Memory cleanup between sessions
        if target_memory_gb <= 8:  # Only for limited memory
            print("🧹 Cleaning memory between sessions...")
            time.sleep(5)  # Give system time to clean up
    
    print(f"\n🎉 Incremental training completed!")
    print(f"📈 Total sessions: {session - 1}")
    print(f"💾 Final model saved to: {model_folder}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Incremental training for limited GPU memory")
    parser.add_argument('-d', '--data_folder', type=str, required=True, help="Path to data folder")
    parser.add_argument('-m', '--model_folder', type=str, required=True, help="Path to model folder")
    parser.add_argument('-e', '--epochs_per_session', type=int, default=5, 
                       help="Maximum epochs per training session (default: 5)")
    parser.add_argument('-mem', '--target_memory_gb', type=float, default=None,
                       help="Target GPU memory usage in GB (auto-detect if not specified)")
    parser.add_argument('--dry_run', action='store_true', 
                       help="Only adjust config without training")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔧 Dry run: Adjusting config only")
        adjust_config_for_memory("config/config.json", args.target_memory_gb or 4)
        return
    
    success = incremental_train(
        args.data_folder, 
        args.model_folder, 
        args.epochs_per_session,
        args.target_memory_gb
    )
    
    if success:
        print("\n🏆 Training completed successfully!")
    else:
        print("\n💥 Training failed or was interrupted")

if __name__ == "__main__":
    main()
