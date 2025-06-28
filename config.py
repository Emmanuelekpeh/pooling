#!/usr/bin/env python3
"""
Centralized configuration management for the Ukiyo-e generation project.
This replaces hardcoded constants and provides environment-based configuration.
"""

import os
import json
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model architectures."""
    # StyleGAN Configuration
    z_dim: int = 64
    w_dim: int = 128
    mapping_layers: int = 8
    
    # NCA Configuration  
    nca_channels: int = 8
    nca_hidden: int = 64
    nca_steps_min: int = 18
    nca_steps_max: int = 38
    
    # Transformer Configuration
    transformer_dim: int = 256
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_patch_size: int = 8
    
    # Discriminator Configuration
    disc_channels: int = 64

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 4
    learning_rate: float = 2e-4
    epochs: int = 250
    
    # Optimizer settings
    beta1: float = 0.0
    beta2: float = 0.99
    epsilon: float = 1e-8
    weight_decay: float = 1e-5
    
    # Learning rate scaling
    gen_lr_scale: float = 0.5
    disc_lr_scale: float = 0.5
    transformer_lr_scale: float = 0.1
    nca_lr_scale: float = 0.5
    
    # Training stability
    gradient_clip: float = 1.0
    value_clamp: float = 10.0
    
    # Progressive training
    transformer_switch_epoch: int = 350
    lr_decay_epoch: int = 200
    lr_decay_factor: float = 0.1

@dataclass
class DataConfig:
    """Configuration for data handling."""
    img_size: int = 64
    data_dir: str = "./data/ukiyo-e-small"
    num_workers: int = 0  # Windows compatibility
    
    # Data augmentation
    horizontal_flip: bool = True
    color_jitter: bool = False
    rotation_degrees: int = 0

@dataclass  
class SystemConfig:
    """Configuration for system resources and paths."""
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = False
    
    # Directory paths
    samples_dir: str = "./samples"
    checkpoint_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    
    # Checkpoint management
    keep_checkpoints: int = 5
    checkpoint_interval: int = 10
    backup_interval: int = 50
    
    # Monitoring
    web_port: int = 5000
    update_interval: int = 100  # batches
    
    # Memory management
    pin_memory: bool = False
    empty_cache_interval: int = 50

@dataclass
class ProjectConfig:
    """Complete project configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    system: SystemConfig
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Resolve device automatically
        if self.system.device == "auto":
            self.system.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adjust batch size for CPU training
        if self.system.device == "cpu" and self.training.batch_size > 4:
            print(f"⚠️  Reducing batch size from {self.training.batch_size} to 4 for CPU training")
            self.training.batch_size = 4
        
        # Create directories
        for dir_path in [self.system.samples_dir, self.system.checkpoint_dir, self.system.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Handle environment-specific checkpoint directory
        if os.path.exists("/app"):  # Fly.io deployment
            self.system.checkpoint_dir = "/app/checkpoints"
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'BATCH_SIZE': ('training', 'batch_size', int),
            'LEARNING_RATE': ('training', 'learning_rate', float),
            'EPOCHS': ('training', 'epochs', int),
            'IMG_SIZE': ('data', 'img_size', int),
            'DATA_DIR': ('data', 'data_dir', str),
            'CHECKPOINT_DIR': ('system', 'checkpoint_dir', str),
            'WEB_PORT': ('system', 'web_port', int),
            'DEVICE': ('system', 'device', str),
            'Z_DIM': ('model', 'z_dim', int),
            'W_DIM': ('model', 'w_dim', int),
        }
        
        for env_key, (section, attr, type_func) in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    typed_value = type_func(env_value)
                    section_obj = getattr(self, section)
                    setattr(section_obj, attr, typed_value)
                    print(f"🔧 Override {section}.{attr} = {typed_value} (from {env_key})")
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Invalid environment variable {env_key}={env_value}: {e}")

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProjectConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclass structure
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        data_config = DataConfig(**config_dict['data'])
        system_config = SystemConfig(**config_dict['system'])
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            system=system_config
        )
    
    def get_device(self) -> torch.device:
        """Get PyTorch device object."""
        return torch.device(self.system.device)
    
    def get_checkpoint_path(self, filename: str = "latest_checkpoint.pt") -> str:
        """Get full path for checkpoint file."""
        return os.path.join(self.system.checkpoint_dir, filename)
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("🚀 PROJECT CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"📊 Model Configuration:")
        print(f"  • Z dimension: {self.model.z_dim}")
        print(f"  • W dimension: {self.model.w_dim}")
        print(f"  • NCA channels: {self.model.nca_channels}")
        print(f"  • NCA steps: {self.model.nca_steps_min}-{self.model.nca_steps_max}")
        
        print(f"\n🎯 Training Configuration:")
        print(f"  • Batch size: {self.training.batch_size}")
        print(f"  • Learning rate: {self.training.learning_rate}")
        print(f"  • Epochs: {self.training.epochs}")
        print(f"  • Device: {self.system.device}")
        
        print(f"\n💾 Data Configuration:")
        print(f"  • Image size: {self.data.img_size}x{self.data.img_size}")
        print(f"  • Data directory: {self.data.data_dir}")
        
        print(f"\n⚙️  System Configuration:")
        print(f"  • Checkpoint directory: {self.system.checkpoint_dir}")
        print(f"  • Web port: {self.system.web_port}")
        print(f"  • Keep checkpoints: {self.system.keep_checkpoints}")
        
        print("=" * 60)


# Global configuration instance
_config: Optional[ProjectConfig] = None

def get_config() -> ProjectConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = create_default_config()
    return _config

def set_config(config: ProjectConfig):
    """Set the global configuration instance."""
    global _config
    _config = config

def create_default_config() -> ProjectConfig:
    """Create default configuration."""
    return ProjectConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
        system=SystemConfig()
    )

def load_config_from_file(filepath: str) -> ProjectConfig:
    """Load and set configuration from file."""
    config = ProjectConfig.load_from_file(filepath)
    set_config(config)
    return config

def load_config_from_env() -> ProjectConfig:
    """Load configuration with environment overrides."""
    config = create_default_config()
    set_config(config)
    return config

# Compatibility functions for existing code
def get_device():
    """Legacy compatibility function."""
    return get_config().get_device()

# Configuration presets
def get_gpu_config() -> ProjectConfig:
    """Configuration optimized for GPU training."""
    config = create_default_config()
    config.training.batch_size = 8
    config.system.mixed_precision = True
    config.system.pin_memory = True
    config.training.learning_rate = 1e-4
    return config

def get_cpu_config() -> ProjectConfig:
    """Configuration optimized for CPU training."""
    config = create_default_config()
    config.training.batch_size = 2
    config.model.nca_channels = 6
    config.model.transformer_dim = 128
    config.model.transformer_depth = 4
    config.training.learning_rate = 3e-4
    return config

def get_debug_config() -> ProjectConfig:
    """Configuration for debugging and testing."""
    config = create_default_config()
    config.training.batch_size = 1
    config.training.epochs = 10
    config.model.nca_steps_min = 5
    config.model.nca_steps_max = 10
    config.system.update_interval = 1
    return config

if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")
    
    # Test default config
    config = create_default_config()
    config.print_summary()
    
    # Test saving/loading
    config.save_to_file("test_config.json")
    loaded_config = ProjectConfig.load_from_file("test_config.json")
    
    print(f"\n✅ Configuration test: {'PASSED' if loaded_config.model.z_dim == config.model.z_dim else 'FAILED'}")
    
    # Clean up
    if os.path.exists("test_config.json"):
        os.remove("test_config.json") 