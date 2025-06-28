#!/usr/bin/env python3
"""
Consolidated Training Runner - Uses existing train_integrated_fast.py but fixes issues
This script imports the working models and functions, then runs a fixed training loop.
"""

import os
import sys
import signal
import time
import json
from pathlib import Path

print("🚀 Loading Consolidated Training Runner...")

# Import everything from the existing working script
try:
    from train_integrated_fast import (
        # Configuration
        Z_DIM, W_DIM, IMG_SIZE, NCA_CHANNELS, BATCH_SIZE, LR, EPOCHS,
        NCA_STEPS_MIN, NCA_STEPS_MAX, DEVICE, DATA_DIR, CHECKPOINT_DIR,
        
        # Models (all working)
        IntegratedGenerator, IntegratedNCA, Discriminator, 
        CrossEvaluator, NCAEvaluator, TransformerCritic,
        
        # Dataset
        ImageDataset,
        
        # Utility functions
        tensor_to_b64, update_status
    )
    print("✅ Imported models and functions from train_integrated_fast.py")
except ImportError as e:
    print(f"❌ Failed to import from train_integrated_fast.py: {e}")
    sys.exit(1)

# Import enhanced features if available
try:
    from enhanced_cross_learning_fixed import EnhancedCrossLearningSystem
    ENHANCED_AVAILABLE = True
    print("✅ Enhanced cross-learning available")
except ImportError:
    ENHANCED_AVAILABLE = False
    print("⚠️  Enhanced cross-learning not available")

try:
    from w_space_stabilizer import WSpaceStabilizer
    W_STABILIZER_AVAILABLE = True
    print("✅ W-space stabilizer available")
except ImportError:
    W_STABILIZER_AVAILABLE = False
    print("⚠️  W-space stabilizer not available")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

def apply_cpu_optimizations():
    """Apply CPU optimizations from PyTorch guide"""
    print("🖥️  Applying CPU optimizations...")
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    torch.set_float32_matmul_precision('high')
    print(f"✅ PyTorch threads: {num_threads}")
    return min(4, num_threads // 2)  # Optimal workers for CPU

def save_checkpoint_fixed(epoch, models_dict, optimizers_dict, checkpoint_dir, scores=None):
    """Fixed checkpoint saving with proper metrics"""
    try:
        checkpoint_data = {
            'epoch': epoch,
            'timestamp': time.time()
        }
        
        # Save model states with the exact keys the loader expects
        for name, model in models_dict.items():
            checkpoint_data[f'{name}_state_dict'] = model.state_dict()
        
        # Save optimizer states
        for name, optimizer in optimizers_dict.items():
            checkpoint_data[f'{name}_state_dict'] = optimizer.state_dict()
        
        # Handle metrics properly - this is what was missing!
        if scores:
            checkpoint_data['scores'] = scores
            # Also save in the format the web interface expects
            if not hasattr(save_checkpoint_fixed, 'metrics_history'):
                save_checkpoint_fixed.metrics_history = []
            save_checkpoint_fixed.metrics_history.append(scores)
            checkpoint_data['metrics_history'] = save_checkpoint_fixed.metrics_history
            print(f"📊 Saved metrics: {list(scores.keys())}")
        
        # Save both latest and epoch checkpoints
        latest_path = Path(checkpoint_dir) / 'latest_checkpoint.pt'
        epoch_path = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint_data, latest_path)
        torch.save(checkpoint_data, epoch_path)
        
        print(f"✅ Checkpoint saved: epoch {epoch}")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_checkpoint_fixed(models_dict, optimizers_dict, checkpoint_dir):
    """Fixed checkpoint loading"""
    latest_path = Path(checkpoint_dir) / 'latest_checkpoint.pt'
    
    if not latest_path.exists():
        print("No checkpoint found, starting fresh")
        return 0
    
    try:
        checkpoint = torch.load(latest_path, weights_only=False, map_location=DEVICE)
        
        # Load model states with error handling for architecture mismatches
        for name, model in models_dict.items():
            state_key = f'{name}_state_dict'
            if state_key in checkpoint:
                try:
                    model.load_state_dict(checkpoint[state_key])
                    print(f"✅ Loaded {name}")
                except RuntimeError as e:
                    if "size mismatch" in str(e) or "Unexpected key(s)" in str(e):
                        print(f"⚠️  {name} architecture changed, starting fresh")
                    else:
                        print(f"❌ Failed to load {name}: {e}")
                        raise
        
        # Load optimizer states
        for name, optimizer in optimizers_dict.items():
            state_key = f'{name}_state_dict'
            if state_key in checkpoint:
                optimizer.load_state_dict(checkpoint[state_key])
                print(f"✅ Loaded {name} optimizer")
        
        # Restore metrics history for web interface
        if 'metrics_history' in checkpoint:
            save_checkpoint_fixed.metrics_history = checkpoint['metrics_history']
            print(f"✅ Restored {len(save_checkpoint_fixed.metrics_history)} epochs of metrics")
        
        epoch = checkpoint.get('epoch', 0)
        print(f"✅ Resumed from epoch {epoch}")
        return epoch
        
    except Exception as e:
        print(f"❌ Checkpoint load failed: {e}")
        import traceback
        traceback.print_exc()
        return 0

def consolidated_training_loop():
    """Main training loop using existing models but with fixes"""
    
    # Setup graceful interruption
    interrupted = False
    
    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n🛑 Graceful shutdown initiated")
        print("Finishing current batch and saving checkpoint...")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Apply CPU optimizations
        cpu_workers = apply_cpu_optimizations()
        
        print("🚀 Starting Consolidated Training")
        print("=" * 60)
        print(f"Device: {DEVICE}")
        print(f"Enhanced Features: {ENHANCED_AVAILABLE}")
        print(f"W-Space Stabilizer: {W_STABILIZER_AVAILABLE}")
        print("=" * 60)
        
        # Initialize all models using existing classes
        print("🏗️  Initializing models...")
        generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
        discriminator = Discriminator(IMG_SIZE).to(DEVICE)
        nca = IntegratedNCA(NCA_CHANNELS, W_DIM).to(DEVICE)
        gen_evaluator = CrossEvaluator(IMG_SIZE).to(DEVICE)
        nca_evaluator = NCAEvaluator(IMG_SIZE).to(DEVICE)
        transformer_critic = TransformerCritic().to(DEVICE)
        
        # Optional enhanced features
        w_stabilizer = None
        cross_learning = None
        
        if W_STABILIZER_AVAILABLE:
            w_stabilizer = WSpaceStabilizer(W_DIM).to(DEVICE)
            print("✅ W-space stabilizer initialized")
        
        if ENHANCED_AVAILABLE:
            cross_learning = EnhancedCrossLearningSystem(IMG_SIZE).to(DEVICE)
            print("✅ Enhanced cross-learning initialized")
        
        # Initialize optimizers
        print("⚙️  Initializing optimizers...")
        gen_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
        nca_optimizer = optim.Adam(nca.parameters(), lr=LR, betas=(0.5, 0.999))
        gen_eval_optimizer = optim.Adam(gen_evaluator.parameters(), lr=LR, betas=(0.5, 0.999))
        nca_eval_optimizer = optim.Adam(nca_evaluator.parameters(), lr=LR, betas=(0.5, 0.999))
        transformer_optimizer = optim.Adam(transformer_critic.parameters(), lr=LR, betas=(0.5, 0.999))
        
        # Organize for checkpoint management
        models_dict = {
            'generator': generator,
            'discriminator': discriminator,
            'nca': nca,
            'gen_evaluator': gen_evaluator,
            'nca_evaluator': nca_evaluator,
            'transformer_critic': transformer_critic
        }
        
        optimizers_dict = {
            'gen_optimizer': gen_optimizer,
            'disc_optimizer': disc_optimizer,
            'nca_optimizer': nca_optimizer,
            'gen_eval_optimizer': gen_eval_optimizer,
            'nca_eval_optimizer': nca_eval_optimizer,
            'transformer_optimizer': transformer_optimizer
        }
        
        if w_stabilizer:
            w_stab_optimizer = optim.Adam(w_stabilizer.parameters(), lr=LR * 0.1)
            models_dict['w_stabilizer'] = w_stabilizer
            optimizers_dict['w_stab_optimizer'] = w_stab_optimizer
        
        if cross_learning:
            cross_optimizer = optim.Adam(cross_learning.parameters(), lr=LR * 0.5)
            models_dict['cross_learning'] = cross_learning
            optimizers_dict['cross_optimizer'] = cross_optimizer
        
        # Initialize dataset
        print("📂 Loading dataset...")
        dataset = ImageDataset(DATA_DIR, IMG_SIZE)
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=cpu_workers,
            persistent_workers=True if cpu_workers > 0 else False,
            pin_memory=False,
            prefetch_factor=2 if cpu_workers > 0 else 2
        )
        print(f"✅ Dataset: {len(dataset)} images, {cpu_workers} workers")
        
        # Load checkpoint if exists
        start_epoch = load_checkpoint_fixed(models_dict, optimizers_dict, CHECKPOINT_DIR) + 1
        
        print(f"🚀 Training from epoch {start_epoch} - Press Ctrl+C to stop gracefully")
        
        # Training loop
        for epoch in range(start_epoch, EPOCHS + 1):
            if interrupted:
                print(f"🛑 Training interrupted at epoch {epoch}")
                break
            
            epoch_start_time = time.time()
            epoch_losses = {
                'gen_loss': 0, 'disc_loss': 0, 'nca_loss': 0,
                'gen_eval_loss': 0, 'nca_eval_loss': 0, 'transformer_loss': 0
            }
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
            
            for batch_idx, real_imgs in enumerate(progress_bar):
                if interrupted:
                    break
                
                real_imgs = real_imgs.to(DEVICE)
                batch_size = real_imgs.shape[0]
                
                # Generate noise
                z = torch.randn(batch_size, Z_DIM, device=DEVICE)
                mixing_z = torch.randn(batch_size, Z_DIM, device=DEVICE) if random.random() < 0.9 else None
                
                # Generate images
                fake_imgs, w = generator(z, return_w=True, mixing_noise=mixing_z)
                
                # Apply W-space stabilization if available
                if w_stabilizer:
                    try:
                        w_variants, w_stats = w_stabilizer(w)
                        # Handle both tensor and dict returns
                        if isinstance(w_variants, dict):
                            w = w_variants.get('stabilized', w)  # Use stabilized version or fallback
                        else:
                            w = w_variants
                    except Exception as e:
                        print(f"⚠️  W-stabilizer error: {e}, using original W")
                        # Continue with original W
                
                # Generate NCA images
                nca_steps = random.randint(NCA_STEPS_MIN, NCA_STEPS_MAX)
                nca_seed = nca.get_seed(batch_size, IMG_SIZE, DEVICE)
                nca_output = nca(nca_seed, w, nca_steps)
                nca_imgs = nca.to_rgb(nca_output)
                
                # Train Discriminator
                disc_optimizer.zero_grad()
                real_pred = discriminator(real_imgs)
                fake_pred = discriminator(fake_imgs.detach())
                nca_pred = discriminator(nca_imgs.detach())
                
                disc_loss = (
                    F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) +
                    F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred)) +
                    F.binary_cross_entropy_with_logits(nca_pred, torch.zeros_like(nca_pred))
                ) / 3
                
                disc_loss.backward()
                disc_optimizer.step()
                
                # Train Generator
                gen_optimizer.zero_grad()
                fake_pred = discriminator(fake_imgs)
                gen_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
                gen_loss.backward()
                gen_optimizer.step()
                
                # Train NCA
                nca_optimizer.zero_grad()
                nca_pred = discriminator(nca_imgs)
                nca_loss = F.binary_cross_entropy_with_logits(nca_pred, torch.ones_like(nca_pred))
                nca_loss.backward()
                nca_optimizer.step()
                
                # Train evaluators
                gen_eval_optimizer.zero_grad()
                gen_eval_loss = F.mse_loss(gen_evaluator(fake_imgs), gen_evaluator(real_imgs))
                gen_eval_loss.backward()
                gen_eval_optimizer.step()
                
                nca_eval_optimizer.zero_grad()
                nca_eval_loss = F.mse_loss(nca_evaluator(nca_imgs), nca_evaluator(real_imgs))
                nca_eval_loss.backward()
                nca_eval_optimizer.step()
                
                # Train transformer critic
                transformer_optimizer.zero_grad()
                transformer_loss = transformer_critic(fake_imgs, real_imgs) + transformer_critic(nca_imgs, real_imgs)
                transformer_loss.backward()
                transformer_optimizer.step()
                
                # Enhanced cross-learning if available
                cross_loss = 0
                if cross_learning:
                    try:
                        cross_optimizer.zero_grad()
                        images_dict = {'generator': fake_imgs, 'nca': nca_imgs, 'real': real_imgs}
                        cross_loss = cross_learning(images_dict, epoch / EPOCHS, batch_idx / len(dataloader))
                        if isinstance(cross_loss, torch.Tensor):
                            cross_loss.backward()
                            cross_optimizer.step()
                            cross_loss = cross_loss.item()
                    except Exception as e:
                        print(f"⚠️  Cross-learning error: {e}")
                        cross_loss = 0
                
                # Accumulate losses
                epoch_losses['gen_loss'] += gen_loss.item()
                epoch_losses['disc_loss'] += disc_loss.item()
                epoch_losses['nca_loss'] += nca_loss.item()
                epoch_losses['gen_eval_loss'] += gen_eval_loss.item()
                epoch_losses['nca_eval_loss'] += nca_eval_loss.item()
                epoch_losses['transformer_loss'] += transformer_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Gen': f"{gen_loss.item():.3f}",
                    'Disc': f"{disc_loss.item():.3f}",
                    'NCA': f"{nca_loss.item():.3f}"
                })
                
                # Update status every 20 batches
                if batch_idx % 20 == 0:
                    status_msg = f"Epoch {epoch}/{EPOCHS}, Batch {batch_idx}/{len(dataloader)}"
                    images = {'real': real_imgs, 'generator': fake_imgs, 'nca': nca_imgs}
                    update_status(status_msg, images=images)
            
            # End of epoch processing
            epoch_time = time.time() - epoch_start_time
            
            # Calculate average losses
            num_batches = len(dataloader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            
            # Create comprehensive scores for checkpoint (matching expected format)
            scores = {
                'gen_eval_loss': epoch_losses['gen_eval_loss'],
                'nca_eval_loss': epoch_losses['nca_eval_loss'],
                'gen_penalty': epoch_losses['gen_loss'],
                'nca_penalty': epoch_losses['nca_loss'],
                'feature_match_gen': epoch_losses['gen_eval_loss'],
                'feature_match_nca': epoch_losses['nca_eval_loss'],
                'transformer_loss': epoch_losses['transformer_loss'],
                'gen_quality': 1.0 - min(epoch_losses['gen_loss'], 1.0),
                'nca_quality': 1.0 - min(epoch_losses['nca_loss'], 1.0),
                'ensemble_prediction': (epoch_losses['gen_eval_loss'] + epoch_losses['nca_eval_loss']) / 2,
                'cross_learning_loss': cross_loss,
                'epoch_time': epoch_time,
                'disc_loss': epoch_losses['disc_loss']
            }
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch}/{EPOCHS} completed in {epoch_time:.2f}s")
            print(f"Gen: {scores['gen_penalty']:.4f}, Disc: {scores['disc_loss']:.4f}, NCA: {scores['nca_penalty']:.4f}")
            print(f"Gen Quality: {scores['gen_quality']:.4f}, NCA Quality: {scores['nca_quality']:.4f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                success = save_checkpoint_fixed(epoch, models_dict, optimizers_dict, CHECKPOINT_DIR, scores)
                if not success:
                    print("⚠️  Checkpoint save failed, but continuing training")
            
            # Update final status for web interface
            status_msg = f"Completed epoch {epoch}/{EPOCHS}"
            images = {'real': real_imgs, 'generator': fake_imgs, 'nca': nca_imgs}
            update_status(status_msg, images=images, scores=scores)
        
        # Final checkpoint save
        final_success = save_checkpoint_fixed(epoch, models_dict, optimizers_dict, CHECKPOINT_DIR, scores)
        
        if final_success:
            print("✅ Training completed successfully!")
            return True
        else:
            print("⚠️  Training completed but final checkpoint failed")
            return False
        
    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        update_status(error_msg, error=True)
        return False

if __name__ == "__main__":
    print("🎯 Consolidated Training Runner")
    print("Using existing models from train_integrated_fast.py with fixes applied")
    print("=" * 60)
    
    # Ensure directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Run consolidated training
    success = consolidated_training_loop()
    
    if success:
        print("🎉 Consolidated training completed successfully!")
    else:
        print("💥 Training failed!")
        sys.exit(1) 