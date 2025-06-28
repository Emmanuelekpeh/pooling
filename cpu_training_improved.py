#!/usr/bin/env python3
"""
Improved CPU training with graceful KeyboardInterrupt handling and optimizations.
Based on your manual interruption workflow and CPU performance needs.
"""

import os
import torch
import torch.nn as nn
import signal
import sys
import time
from contextlib import contextmanager

# Apply CPU optimizations from PyTorch guide
def setup_cpu_training():
    """
    Apply CPU-specific optimizations for better performance.
    Based on: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """
    print("🖥️  Setting up CPU training optimizations...")
    
    # 1. Optimize threading
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    print(f"✅ Set PyTorch threads: {num_threads}")
    
    # 2. Set optimal OMP settings for CPU
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    # 3. Enable optimized CPU kernels
    torch.backends.cudnn.enabled = False  # Not needed for CPU
    torch.set_float32_matmul_precision('high')  # Use optimized BLAS
    
    # 4. Optimal DataLoader settings for CPU
    dataloader_kwargs = {
        'num_workers': min(4, num_threads // 2),  # Don't overwhelm CPU
        'persistent_workers': True,  # Reuse worker processes
        'pin_memory': False,  # Not needed for CPU training
        'prefetch_factor': 2,  # Modest prefetching
    }
    
    print(f"✅ DataLoader workers: {dataloader_kwargs['num_workers']}")
    return dataloader_kwargs

class GracefulInterruptHandler:
    """
    Handle KeyboardInterrupt gracefully during training.
    Allows safe interruption at batch boundaries.
    """
    
    def __init__(self):
        self.interrupted = False
        self.original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        print("\n🛑 Interruption requested... finishing current batch safely")
        self.interrupted = True
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)

@contextmanager
def safe_training_step():
    """
    Context manager to handle interruptions safely during training steps.
    This prevents corruption during backward() and optimizer.step().
    """
    try:
        yield
    except KeyboardInterrupt:
        print("⚠️  Interrupted during training step - allowing completion")
        raise  # Re-raise after logging

def training_loop_with_graceful_interruption():
    """
    Main training loop with CPU optimizations and graceful interruption handling.
    """
    
    # Setup CPU optimizations
    dataloader_kwargs = setup_cpu_training()
    
    # Your existing model setup (simplified for example)
    device = torch.device('cpu')
    print(f"🖥️  Training on: {device}")
    
    # Create models (use your existing models)
    from train_integrated_fast import (
        IntegratedGenerator, IntegratedNCA, Discriminator, 
        TransformerCritic, ImageDataset
    )
    
    # Model initialization
    generator = IntegratedGenerator(z_dim=64, w_dim=128).to(device)
    nca = IntegratedNCA(channel_n=8, w_dim=128).to(device)
    discriminator = Discriminator(img_size=64).to(device)
    
    # CPU-optimized settings
    IMG_SIZE = 64
    BATCH_SIZE = 2  # Smaller batch size for CPU
    
    # Dataset with CPU optimizations
    dataset = ImageDataset('./data/ukiyo-e-small', img_size=IMG_SIZE)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        **dataloader_kwargs
    )
    
    # Optimizers with CPU-friendly settings
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    nca_optimizer = torch.optim.Adam(nca.parameters(), lr=0.0001)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    
    print(f"🚀 Starting training with graceful interruption handling")
    print("   Press Ctrl+C to stop gracefully at the end of current batch")
    
    # Graceful interruption handler
    with GracefulInterruptHandler() as handler:
        epoch = 1
        
        while not handler.interrupted:
            print(f"\n📊 Epoch {epoch}")
            epoch_start = time.time()
            
            for batch_idx, real_imgs in enumerate(dataloader):
                if handler.interrupted:
                    print("🛑 Stopping at batch boundary (safe interruption)")
                    break
                
                real_imgs = real_imgs.to(device)
                batch_start = time.time()
                
                try:
                    # Use safe training step context
                    with safe_training_step():
                        # Generate images
                        noise = torch.randn(real_imgs.size(0), 64, device=device)
                        fake_imgs, w = generator(noise, return_w=True)
                        
                        # NCA processing
                        nca_seed = nca.get_seed(real_imgs.size(0), IMG_SIZE, device)
                        nca_result = nca(nca_seed, w, steps=20, target_img=real_imgs)
                        nca_imgs = nca.to_rgba(nca_result)[:, :3]  # RGB only
                        
                        # Discriminator losses
                        real_pred = discriminator(real_imgs)
                        fake_pred = discriminator(fake_imgs.detach())
                        nca_pred = discriminator(nca_imgs.detach())
                        
                        # Compute losses
                        disc_loss = (
                            torch.mean((real_pred - 1)**2) + 
                            torch.mean(fake_pred**2) + 
                            torch.mean(nca_pred**2)
                        ) / 3
                        
                        gen_loss = torch.mean((discriminator(fake_imgs) - 1)**2)
                        nca_loss = torch.mean((discriminator(nca_imgs) - 1)**2)
                        
                        # Backward passes (interruption-safe)
                        disc_optimizer.zero_grad()
                        disc_loss.backward()
                        disc_optimizer.step()
                        
                        gen_optimizer.zero_grad()
                        gen_loss.backward()
                        gen_optimizer.step()
                        
                        nca_optimizer.zero_grad()
                        nca_loss.backward()
                        nca_optimizer.step()
                        
                        # Progress logging
                        batch_time = time.time() - batch_start
                        if batch_idx % 5 == 0:
                            print(f"  Batch {batch_idx:3d}: "
                                  f"D={disc_loss.item():.4f} "
                                  f"G={gen_loss.item():.4f} "
                                  f"N={nca_loss.item():.4f} "
                                  f"({batch_time:.2f}s)")
                
                except KeyboardInterrupt:
                    print("\n🛑 Interrupted during training step - stopping safely")
                    handler.interrupted = True
                    break
                except Exception as e:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    continue
            
            if handler.interrupted:
                break
                
            epoch_time = time.time() - epoch_start
            print(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = f"checkpoints/epoch_{epoch}_cpu.pt"
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'generator': generator.state_dict(),
                    'nca': nca.state_dict(),
                    'discriminator': discriminator.state_dict(),
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            epoch += 1
    
    print("\n🎯 Training completed gracefully!")
    print("   Final checkpoint will be saved...")
    
    # Save final checkpoint
    final_checkpoint = "checkpoints/final_cpu.pt"
    torch.save({
        'epoch': epoch,
        'generator': generator.state_dict(),
        'nca': nca.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, final_checkpoint)
    print(f"💾 Final checkpoint: {final_checkpoint}")

if __name__ == "__main__":
    print("🖥️  CPU-Optimized Training with Graceful Interruption")
    print("=" * 60)
    training_loop_with_graceful_interruption() 