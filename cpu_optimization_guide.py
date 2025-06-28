#!/usr/bin/env python3
"""
CPU Optimization Guide for Ukiyo-e Training
Based on PyTorch Performance Tuning Guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import get_config

def apply_cpu_optimizations():
    """
    Apply CPU-specific optimizations from PyTorch tuning guide.
    Call this at the start of your training script.
    """
    
    print("🖥️  Applying CPU optimizations...")
    
    # 1. Set optimal number of threads
    # PyTorch guide: "num_workers should be tuned depending on the workload, CPU"
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    print(f"✅ Set PyTorch threads to {num_threads}")
    
    # 2. Enable OpenMP optimizations
    # From PyTorch guide: "Utilize OpenMP"
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    print(f"✅ Set OpenMP threads to {num_threads}")
    
    # 3. Use Intel OpenMP if available (better performance)
    # PyTorch guide: "Intel OpenMP Runtime Library (libiomp)"
    try:
        import intel_extension_for_pytorch as ipex
        print("✅ Intel Extension for PyTorch available")
    except ImportError:
        print("ℹ️  Intel Extension not available (install with: pip install intel_extension_for_pytorch)")
    
    # 4. Switch memory allocator for better performance
    # PyTorch guide: "Switch Memory allocator"
    os.environ['PYTORCH_MEM_ALLOCATOR'] = 'native'
    print("✅ Using native memory allocator")
    
    # 5. Enable JIT optimizations
    torch.jit.enable_onednn_fusion(True)
    print("✅ Enabled oneDNN fusion")
    
    # 6. Set optimal data loading parameters for CPU
    return {
        'num_workers': min(4, num_threads // 2),  # Don't overwhelm with too many workers
        'pin_memory': False,  # Not needed for CPU training
        'persistent_workers': True,  # Reduce worker startup overhead
    }

def optimize_dataloader_for_cpu(dataset, batch_size=4):
    """
    Create optimized DataLoader for CPU training.
    Based on PyTorch guide: "Enable asynchronous data loading and augmentation"
    """
    
    dataloader_kwargs = apply_cpu_optimizations()
    
    # Your current issue: KeyboardInterrupt during data loading
    # Fix: Use fewer workers and enable persistent workers
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs,
        # Add error handling for data loading
        worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id)
    )
    
    print(f"✅ Optimized DataLoader: batch_size={batch_size}, workers={dataloader_kwargs['num_workers']}")
    return dataloader

def optimize_model_for_cpu(model):
    """
    Apply CPU-specific model optimizations.
    """
    
    print("🔧 Applying CPU model optimizations...")
    
    # 1. Use inference mode for evaluation (PyTorch guide)
    model.eval()
    
    # 2. Disable gradient computation for validation
    # PyTorch guide: "Disable gradient calculation for validation or inference"
    @torch.no_grad()
    def inference_wrapper(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    # 3. Use TorchScript for inference speedup
    # PyTorch guide: "Use oneDNN Graph with TorchScript for inference"
    try:
        model_traced = torch.jit.trace(model, torch.randn(1, 3, 64, 64))
        print("✅ Model traced with TorchScript")
        return model_traced
    except Exception as e:
        print(f"⚠️  TorchScript tracing failed: {e}")
        return model

def optimize_training_for_cpu():
    """
    CPU-specific training optimizations.
    """
    
    optimizations = {
        # Reduce batch size for CPU (current: 4 is good)
        'batch_size': 4,
        
        # Use smaller model dimensions for CPU
        'reduce_model_size': True,
        
        # Gradient accumulation to simulate larger batches
        'gradient_accumulation_steps': 4,  # Effective batch size = 4 * 4 = 16
        
        # Less frequent checkpointing to reduce I/O
        'checkpoint_interval': 20,  # Every 20 epochs instead of 10
        
        # Disable expensive operations
        'disable_amp': True,  # Mixed precision not beneficial on CPU
        'disable_compile': True,  # torch.compile targets GPU
    }
    
    return optimizations

def create_cpu_optimized_config():
    """
    Create a configuration optimized for CPU training.
    Addresses your current training issues.
    """
    
    from config import create_default_config
    
    config = create_default_config()
    
    # CPU-specific model adjustments
    config.model.z_dim = 32  # Reduced from 64
    config.model.w_dim = 64  # Reduced from 128
    config.model.nca_channels = 6  # Reduced from 8
    config.model.nca_hidden = 32  # Reduced from 64
    config.model.transformer_dim = 128  # Reduced from 256
    config.model.transformer_depth = 4  # Reduced from 6
    
    # CPU-specific training adjustments
    config.training.batch_size = 4  # Keep current
    config.training.learning_rate = 3e-4  # Slightly higher for smaller batches
    config.training.gradient_clip = 0.5  # Tighter clipping for stability
    
    # CPU-specific system settings
    config.system.device = "cpu"
    config.system.mixed_precision = False
    config.system.pin_memory = False
    config.data.num_workers = min(4, os.cpu_count() // 2)
    
    # Reduce frequency of expensive operations
    config.system.update_interval = 50  # Less frequent updates
    config.system.checkpoint_interval = 20  # Less frequent checkpoints
    
    return config

def fix_current_training_issues():
    """
    Specific fixes for the issues shown in your logs.
    """
    
    fixes = {
        'keyboard_interrupt_issue': {
            'problem': 'Training stops on KeyboardInterrupt during backward pass',
            'solution': 'Add proper signal handling and graceful shutdown',
            'code': '''
try:
    # training loop
    for epoch in range(start_epoch, epochs):
        for batch_idx, data in enumerate(dataloader):
            # training code
            loss.backward()
            optimizer.step()
except KeyboardInterrupt:
    print("Training interrupted - saving checkpoint...")
    save_checkpoint_safely(epoch, models, optimizers)
    print("Checkpoint saved. Training can be resumed.")
    exit(0)
            '''
        },
        
        'data_loading_interrupt': {
            'problem': 'KeyboardInterrupt during image transform/resize',
            'solution': 'Use persistent workers and error handling',
            'code': '''
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,  # Reduced from 4
    persistent_workers=True,  # Keep workers alive
    timeout=30,  # Add timeout
)
            '''
        },
        
        'checkpoint_corruption': {
            'problem': 'Checkpoint corruption causing training restarts',
            'solution': 'Use the RobustCheckpointManager created earlier',
            'code': '''
from checkpoint_fix import RobustCheckpointManager

checkpoint_manager = RobustCheckpointManager("./checkpoints")

# Save with atomic writes
success = checkpoint_manager.save_checkpoint(epoch, checkpoint_data)

# Load with fallback recovery
checkpoint = checkpoint_manager.load_checkpoint()
            '''
        }
    }
    
    return fixes

def main():
    """
    Main function to test CPU optimizations.
    """
    
    print("🖥️  CPU Optimization Guide for Ukiyo-e Training")
    print("=" * 60)
    
    # Apply optimizations
    dataloader_kwargs = apply_cpu_optimizations()
    print(f"\n📊 Recommended DataLoader settings: {dataloader_kwargs}")
    
    # Show optimized configuration
    config = create_cpu_optimized_config()
    print(f"\n🔧 CPU-Optimized Configuration:")
    print(f"  • Model size reduced: Z={config.model.z_dim}, W={config.model.w_dim}")
    print(f"  • NCA channels: {config.model.nca_channels}")
    print(f"  • Transformer: dim={config.model.transformer_dim}, depth={config.model.transformer_depth}")
    print(f"  • Data workers: {config.data.num_workers}")
    
    # Show fixes for current issues
    fixes = fix_current_training_issues()
    print(f"\n🔧 Fixes for Current Issues:")
    for issue, details in fixes.items():
        print(f"  • {details['problem']}")
        print(f"    Solution: {details['solution']}")
    
    print("\n✅ Ready to apply CPU optimizations!")

if __name__ == "__main__":
    main() 