#!/usr/bin/env python3
"""
Robust checkpoint management system to fix corruption issues.
This addresses the critical training interruption problems.
"""

import os
import torch
import tempfile
import shutil
from typing import Dict, Any, Optional
import traceback
import json
import time

class RobustCheckpointManager:
    """
    A robust checkpoint management system that prevents corruption
    and provides recovery mechanisms.
    """
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.backup_dir = os.path.join(checkpoint_dir, 'backups')
        
        # Ensure directories exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def save_checkpoint(self, epoch: int, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Save checkpoint with atomic write to prevent corruption.
        Returns True if successful, False otherwise.
        """
        try:
            # Create temporary file for atomic write
            checkpoint_filename = f'checkpoint_epoch_{epoch}.pt'
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            # Use temporary file to ensure atomic write
            with tempfile.NamedTemporaryFile(
                dir=self.checkpoint_dir, 
                delete=False, 
                suffix='.tmp'
            ) as tmp_file:
                temp_path = tmp_file.name
                
                # Save to temporary file
                torch.save(checkpoint_data, temp_path)
                
                # Verify the file was saved correctly
                if self._verify_checkpoint(temp_path):
                    # Atomic move to final location
                    shutil.move(temp_path, checkpoint_path)
                    
                    # Update latest checkpoint atomically
                    self._update_latest_checkpoint(checkpoint_path)
                    
                    # Create backup of important checkpoints
                    if epoch % 10 == 0:  # Backup every 10 epochs
                        self._create_backup(epoch, checkpoint_path)
                    
                    # Clean up old checkpoints
                    self._cleanup_old_checkpoints()
                    
                    print(f"✅ Checkpoint saved successfully at epoch {epoch}")
                    return True
                else:
                    # Remove corrupted temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    print(f"❌ Checkpoint verification failed at epoch {epoch}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error saving checkpoint at epoch {epoch}: {str(e)}")
            traceback.print_exc()
            
            # Clean up temporary file if it exists
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
                
            return False
    
    def _verify_checkpoint(self, checkpoint_path: str) -> bool:
        """Verify that a checkpoint file is valid and loadable."""
        try:
            # Try to load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Basic validation
            required_keys = ['epoch', 'generator_state_dict', 'discriminator_state_dict']
            for key in required_keys:
                if key not in checkpoint:
                    print(f"Missing required key: {key}")
                    return False
            
            # Check that state dicts are not empty
            if not checkpoint['generator_state_dict']:
                print("Generator state dict is empty")
                return False
                
            return True
            
        except Exception as e:
            print(f"Checkpoint verification failed: {str(e)}")
            return False
    
    def _update_latest_checkpoint(self, checkpoint_path: str):
        """Update latest checkpoint link atomically."""
        try:
            latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
            temp_latest = latest_path + '.tmp'
            
            # Copy to temporary file first
            shutil.copy2(checkpoint_path, temp_latest)
            
            # Atomic move
            shutil.move(temp_latest, latest_path)
            
        except Exception as e:
            print(f"Warning: Failed to update latest checkpoint: {str(e)}")
    
    def _create_backup(self, epoch: int, checkpoint_path: str):
        """Create a backup of important checkpoints."""
        try:
            backup_filename = f'backup_epoch_{epoch}_{int(time.time())}.pt'
            backup_path = os.path.join(self.backup_dir, backup_filename)
            shutil.copy2(checkpoint_path, backup_path)
            print(f"📦 Created backup: {backup_filename}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {str(e)}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping the most recent ones."""
        try:
            # Get all epoch checkpoint files
            checkpoint_files = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                    try:
                        epoch_str = filename.replace('checkpoint_epoch_', '').replace('.pt', '')
                        epoch_num = int(epoch_str)
                        filepath = os.path.join(self.checkpoint_dir, filename)
                        checkpoint_files.append((epoch_num, filepath))
                    except ValueError:
                        continue
            
            # Sort by epoch number (newest first)
            checkpoint_files.sort(key=lambda x: x[0], reverse=True)
            
            # Remove old checkpoints beyond keep_last_n
            if len(checkpoint_files) > self.keep_last_n:
                files_to_remove = checkpoint_files[self.keep_last_n:]
                for epoch_num, filepath in files_to_remove:
                    try:
                        os.remove(filepath)
                        print(f"🗑️  Removed old checkpoint: epoch_{epoch_num}")
                    except Exception as e:
                        print(f"Warning: Could not remove {filepath}: {str(e)}")
                        
        except Exception as e:
            print(f"Warning: Checkpoint cleanup failed: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint with fallback recovery mechanisms.
        Returns None if no valid checkpoint is found.
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        
        # Try to load the specified checkpoint
        checkpoint = self._try_load_single_checkpoint(checkpoint_path)
        if checkpoint is not None:
            return checkpoint
        
        print(f"⚠️  Primary checkpoint failed, trying recovery...")
        
        # Try to find the most recent valid checkpoint
        checkpoint = self._find_most_recent_valid_checkpoint()
        if checkpoint is not None:
            print("✅ Recovered from backup checkpoint")
            return checkpoint
        
        # Try backup directory
        checkpoint = self._try_backup_recovery()
        if checkpoint is not None:
            print("✅ Recovered from backup directory")
            return checkpoint
        
        print("❌ No valid checkpoints found")
        return None
    
    def _try_load_single_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Try to load a single checkpoint file."""
        try:
            if not os.path.exists(checkpoint_path):
                return None
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Verify checkpoint integrity
            if self._verify_checkpoint_data(checkpoint):
                return checkpoint
            else:
                print(f"⚠️  Checkpoint {checkpoint_path} failed integrity check")
                return None
                
        except Exception as e:
            print(f"⚠️  Failed to load {checkpoint_path}: {str(e)}")
            return None
    
    def _verify_checkpoint_data(self, checkpoint: Dict[str, Any]) -> bool:
        """Verify loaded checkpoint data integrity."""
        try:
            required_keys = ['epoch', 'generator_state_dict', 'discriminator_state_dict']
            
            for key in required_keys:
                if key not in checkpoint:
                    return False
            
            # Verify epoch is valid
            if not isinstance(checkpoint['epoch'], int) or checkpoint['epoch'] < 0:
                return False
            
            # Verify state dicts are not empty
            if not checkpoint['generator_state_dict']:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _find_most_recent_valid_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Find the most recent valid checkpoint in the main directory."""
        try:
            checkpoint_files = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                    try:
                        epoch_str = filename.replace('checkpoint_epoch_', '').replace('.pt', '')
                        epoch_num = int(epoch_str)
                        filepath = os.path.join(self.checkpoint_dir, filename)
                        checkpoint_files.append((epoch_num, filepath))
                    except ValueError:
                        continue
            
            # Sort by epoch (newest first)
            checkpoint_files.sort(key=lambda x: x[0], reverse=True)
            
            # Try each checkpoint until we find a valid one
            for epoch_num, filepath in checkpoint_files:
                print(f"🔍 Trying checkpoint from epoch {epoch_num}...")
                checkpoint = self._try_load_single_checkpoint(filepath)
                if checkpoint is not None:
                    return checkpoint
            
            return None
            
        except Exception as e:
            print(f"Error during checkpoint recovery: {str(e)}")
            return None
    
    def _try_backup_recovery(self) -> Optional[Dict[str, Any]]:
        """Try to recover from backup directory."""
        try:
            if not os.path.exists(self.backup_dir):
                return None
            
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith('backup_epoch_') and filename.endswith('.pt'):
                    filepath = os.path.join(self.backup_dir, filename)
                    backup_files.append(filepath)
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Try each backup until we find a valid one
            for filepath in backup_files:
                print(f"🔍 Trying backup: {os.path.basename(filepath)}...")
                checkpoint = self._try_load_single_checkpoint(filepath)
                if checkpoint is not None:
                    return checkpoint
            
            return None
            
        except Exception as e:
            print(f"Error during backup recovery: {str(e)}")
            return None


def create_checkpoint_data(epoch: int, models: Dict, optimizers: Dict, 
                          metrics_history: list = None) -> Dict[str, Any]:
    """Helper function to create checkpoint data dictionary."""
    checkpoint_data = {
        'epoch': epoch,
        'timestamp': time.time(),
        'scores': metrics_history or []
    }
    
    # Add model state dicts
    for name, model in models.items():
        checkpoint_data[f'{name}_state_dict'] = model.state_dict()
    
    # Add optimizer state dicts
    for name, optimizer in optimizers.items():
        checkpoint_data[f'{name}_state_dict'] = optimizer.state_dict()
    
    return checkpoint_data


# Usage example integration
def integrate_robust_checkpoints():
    """
    Example of how to integrate this into the training loop.
    Replace the existing save_checkpoint function with this approach.
    """
    
    # Initialize checkpoint manager
    checkpoint_manager = RobustCheckpointManager(
        checkpoint_dir=os.environ.get("CHECKPOINT_DIR", "./checkpoints"),
        keep_last_n=5
    )
    
    # In training loop:
    def save_training_checkpoint(epoch, generator, discriminator, gen_evaluator, 
                               nca_evaluator, transformer_critic, nca_model,
                               gen_optimizer, disc_optimizer, transformer_optimizer, 
                               nca_optimizer, metrics_history=None):
        
        models = {
            'generator': generator,
            'discriminator': discriminator,
            'gen_evaluator': gen_evaluator,
            'nca_evaluator': nca_evaluator,
            'transformer_critic': transformer_critic,
            'nca_model': nca_model
        }
        
        optimizers = {
            'gen_optimizer': gen_optimizer,
            'disc_optimizer': disc_optimizer,
            'transformer_optimizer': transformer_optimizer,
            'nca_optimizer': nca_optimizer
        }
        
        checkpoint_data = create_checkpoint_data(epoch, models, optimizers, metrics_history)
        
        success = checkpoint_manager.save_checkpoint(epoch, checkpoint_data)
        if not success:
            print(f"⚠️  Failed to save checkpoint at epoch {epoch}")
        
        return success
    
    # Loading checkpoints:
    def load_training_checkpoint(generator, discriminator, gen_evaluator, 
                               nca_evaluator, transformer_critic, nca_model,
                               gen_optimizer, disc_optimizer, transformer_optimizer, 
                               nca_optimizer):
        
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint is None:
            print("No checkpoint found, starting from scratch")
            return 1, []
        
        try:
            # Load model states
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            gen_evaluator.load_state_dict(checkpoint['gen_evaluator_state_dict'])
            nca_evaluator.load_state_dict(checkpoint['nca_evaluator_state_dict'])
            transformer_critic.load_state_dict(checkpoint['transformer_critic_state_dict'])
            nca_model.load_state_dict(checkpoint['nca_model_state_dict'])
            
            # Load optimizer states
            gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer_state_dict'])
            nca_optimizer.load_state_dict(checkpoint['nca_optimizer_state_dict'])
            
            epoch = checkpoint['epoch']
            metrics_history = checkpoint.get('scores', [])
            
            print(f"✅ Loaded checkpoint from epoch {epoch}")
            return epoch + 1, metrics_history
            
        except Exception as e:
            print(f"❌ Error loading checkpoint state: {str(e)}")
            return 1, []

if __name__ == "__main__":
    # Test the checkpoint manager
    print("Testing RobustCheckpointManager...")
    
    manager = RobustCheckpointManager("./test_checkpoints", keep_last_n=3)
    
    # Test saving
    test_data = {
        'epoch': 1,
        'generator_state_dict': {'weight': torch.randn(10, 10)},
        'discriminator_state_dict': {'weight': torch.randn(5, 5)},
        'scores': [{'loss': 0.5}]
    }
    
    success = manager.save_checkpoint(1, test_data)
    print(f"Save test: {'✅ PASSED' if success else '❌ FAILED'}")
    
    # Test loading
    loaded = manager.load_checkpoint()
    load_success = loaded is not None and loaded['epoch'] == 1
    print(f"Load test: {'✅ PASSED' if load_success else '❌ FAILED'}")
    
    print("Checkpoint manager test complete!") 