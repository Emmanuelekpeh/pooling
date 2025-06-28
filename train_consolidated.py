#!/usr/bin/env python3
"""
Consolidated Training Script - Ukiyo-e Generative Art System
Combines all working features from multiple training scripts into one robust system.
"""

import os
import io
import json
import time
import base64
import random
import signal
import sys
from threading import Thread
import math
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from flask import Flask, jsonify, render_template_string

# Import enhanced systems if available
try:
    from enhanced_cross_learning_fixed import SignalWeightingNetwork, EnhancedCrossLearningSystem
    ENHANCED_FEATURES = True
    print("✅ Enhanced cross-learning features loaded")
except ImportError:
    ENHANCED_FEATURES = False
    print("⚠️  Enhanced features not available, using basic training")

try:
    from w_space_stabilizer import WSpaceStabilizer
    W_STABILIZER_AVAILABLE = True
    print("✅ W-space stabilizer loaded")
except ImportError:
    W_STABILIZER_AVAILABLE = False
    print("⚠️  W-space stabilizer not available")

# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration management"""
    # Model dimensions
    Z_DIM = 64
    W_DIM = 128
    IMG_SIZE = 64
    NCA_CHANNELS = 8
    
    # Training parameters
    BATCH_SIZE = 4
    LR = 2e-4
    EPOCHS = 250
    NCA_STEPS_MIN = 18
    NCA_STEPS_MAX = 38
    
    # Paths
    DATA_DIR = "./data/ukiyo-e-small"
    SAMPLES_DIR = "./samples"
    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
    KEEP_CHECKPOINTS = 5
    
    # Device and CPU optimizations
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def apply_cpu_optimizations(cls):
        """Apply CPU-specific optimizations"""
        if cls.DEVICE.type == "cpu":
            print("🖥️  Applying CPU optimizations...")
            num_threads = os.cpu_count()
            torch.set_num_threads(num_threads)
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            torch.set_float32_matmul_precision('high')
            print(f"✅ PyTorch threads: {num_threads}")
            return min(4, num_threads // 2)  # Optimal workers for CPU
        return 0

print("🚀 Consolidated Training Script Loaded")
print(f"Enhanced Features: {ENHANCED_FEATURES}")
print(f"W-Space Stabilizer: {W_STABILIZER_AVAILABLE}")

# ==================== MODELS ====================
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (2 / in_dim) ** 0.5
        
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = (2 / (in_channels * kernel_size ** 2)) ** 0.5
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, 
                       stride=self.stride, padding=self.padding)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = EqualizedLinear(w_dim, channels)
        self.style_bias = EqualizedLinear(w_dim, channels)
        
    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return x * (style_scale + 1) + style_bias

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.pixel_norm = PixelNorm()
        
        layers = []
        in_dim = z_dim
        for _ in range(8):
            layers.extend([
                EqualizedLinear(in_dim, 512),
                nn.LeakyReLU(0.2)
            ])
            in_dim = 512
        
        layers.append(EqualizedLinear(512, w_dim))
        self.mapping = nn.Sequential(*layers)
        
        self.truncation_psi = 0.7
        self.register_buffer('w_avg', torch.zeros(w_dim))
        
    def forward(self, z, truncation=True):
        x = self.pixel_norm(z)
        w = self.mapping(x)
        
        if self.training:
            self.w_avg.copy_(w.detach().mean(0).lerp(self.w_avg, 0.995))
        
        if truncation:
            w = self.w_avg.lerp(w, self.truncation_psi)
            
        return w

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)
        
        self.noise_weight1 = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.01)
        self.noise_weight2 = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.01)
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, w):
        x = self.conv1(x)
        noise1 = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_weight1 * noise1
        x = self.activation(x)
        x = self.adain1(x, w)
        x = torch.clamp(x, -10.0, 10.0)
        
        x = self.conv2(x)
        noise2 = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_weight2 * noise2
        x = self.activation(x)
        x = self.adain2(x, w)
        x = torch.clamp(x, -10.0, 10.0)
        
        return x

class IntegratedGenerator(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, 256, 4, 4) * 0.1)
        
        self.blocks = nn.ModuleList([
            GeneratorBlock(256, 128, w_dim),
            GeneratorBlock(128, 64, w_dim),
            GeneratorBlock(64, 32, w_dim),
            GeneratorBlock(32, 16, w_dim),
        ])
        
        self.to_rgb = nn.ModuleList([
            EqualizedConv2d(128, 3, 1),
            EqualizedConv2d(64, 3, 1),
            EqualizedConv2d(32, 3, 1),
            EqualizedConv2d(16, 3, 1),
        ])
        
        self.register_buffer('alpha', torch.tensor(1.0))
        self.current_block = len(self.blocks) - 1
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, EqualizedConv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, noise, return_w=False, mixing_noise=None):
        w = self.mapping(noise)
        
        if mixing_noise is not None and random.random() < 0.9:
            w2 = self.mapping(mixing_noise)
            cutoff = random.randint(1, len(self.blocks))
            w = [w if i < cutoff else w2 for i in range(len(self.blocks))]
        else:
            w = [w] * len(self.blocks)
        
        x = self.const.repeat(noise.shape[0], 1, 1, 1)
        
        for i, (block, to_rgb_layer, w_layer) in enumerate(zip(self.blocks, self.to_rgb, w)):
            if i <= self.current_block:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = block(x, w_layer)
                
                if i == self.current_block:
                    rgb = to_rgb_layer(x)
                    rgb = torch.tanh(rgb)
        
        if return_w:
            return rgb, w[0] if not isinstance(w[0], list) else w[0][0]
        return rgb

class IntegratedNCA(nn.Module):
    def __init__(self, channel_n, w_dim, hidden_n=64):
        super().__init__()
        self.channel_n = channel_n
        self.w_dim = w_dim
        self.hidden_n = hidden_n
        
        # Perception network
        self.perceive = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, hidden_n, 1),
            nn.ReLU()
        )
        
        # Update network with W conditioning
        self.update_net = nn.Sequential(
            nn.Linear(hidden_n + w_dim, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, channel_n, bias=False)
        )
        
        # Initialize Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        
        filters = torch.stack([identity, sobel_x, sobel_y], dim=0)
        filters = filters.unsqueeze(1).repeat(channel_n, 1, 1, 1)
        
        self.register_buffer('filters', filters)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_seed(self, batch_size, size, device, seed_type="distributed"):
        """Generate initial seed for NCA"""
        seed = torch.zeros(batch_size, self.channel_n, size, size, device=device)
        
        if seed_type == "center":
            # Single seed at center
            center = size // 2
            radius = 2
            y, x = torch.meshgrid(torch.arange(size, device=device), 
                                torch.arange(size, device=device), indexing='ij')
            mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
            seed[:, :3, mask] = torch.rand(batch_size, 3, 1, device=device).expand(-1, -1, mask.sum())
            seed[:, 3, mask] = 1.0
        
        elif seed_type == "distributed":
            # Multiple distributed seeds
            num_seeds = 5
            positions = self._get_distributed_positions(size, num_seeds)
            for center_x, center_y in positions:
                seed = self._place_seed_at(seed, center_x, center_y, 2, batch_size, device)
        
        return seed
    
    def _get_distributed_positions(self, size, num_seeds):
        """Get distributed seed positions"""
        positions = []
        margin = size // 6
        
        # Corner positions
        positions.extend([
            (margin, margin),
            (size - margin, margin),
            (margin, size - margin),
            (size - margin, size - margin),
        ])
        
        # Center position
        positions.append((size // 2, size // 2))
        
        return positions[:num_seeds]
    
    def _place_seed_at(self, seed, center_x, center_y, radius, batch_size, device):
        """Place a seed at specific coordinates"""
        size = seed.shape[-1]
        y, x = torch.meshgrid(torch.arange(size, device=device), 
                            torch.arange(size, device=device), indexing='ij')
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        
        if mask.sum() > 0:
            seed[:, :3, mask] = torch.rand(batch_size, 3, 1, device=device).expand(-1, -1, mask.sum())
            seed[:, 3, mask] = 1.0
        
        return seed
    
    def perceive_environment(self, x):
        """Perceive the environment using convolution filters"""
        y = F.conv2d(x, self.filters, padding=1, groups=self.channel_n)
        y = y.view(x.shape[0], self.channel_n, 3, x.shape[2], x.shape[3])
        y = y.permute(0, 1, 3, 4, 2).contiguous()
        y = y.view(x.shape[0], self.channel_n * 3, x.shape[2], x.shape[3])
        return y
    
    def to_rgb(self, x):
        """Convert NCA state to RGB"""
        rgb = x[:, :3, :, :]
        return torch.tanh(rgb)
    
    def to_rgba(self, x):
        """Convert NCA state to RGBA"""
        rgba = x[:, :4, :, :]
        rgba[:, :3] = torch.tanh(rgba[:, :3])
        rgba[:, 3:4] = torch.sigmoid(rgba[:, 3:4])
        return rgba
    
    def forward(self, x, w, steps, target_img=None):
        """Forward pass with W conditioning"""
        for step in range(steps):
            # Get alive mask
            alive_mask = (x[:, 3:4, :, :] > 0.01).float()
            
            # Perceive environment
            perceived = self.perceive_environment(x)
            perceived = self.perceive(perceived)
            
            # Flatten for linear layers
            b, c, h, w = perceived.shape
            perceived_flat = perceived.view(b, c, -1).permute(0, 2, 1).contiguous()
            perceived_flat = perceived_flat.view(-1, c)
            
            # Expand W for spatial conditioning
            w_expanded = w.unsqueeze(1).expand(-1, h * w, -1).contiguous()
            w_flat = w_expanded.view(-1, self.w_dim)
            
            # Concatenate perception and W
            update_input = torch.cat([perceived_flat, w_flat], dim=1)
            
            # Compute updates
            ds = self.update_net(update_input)
            ds = ds.view(b, h, w, self.channel_n).permute(0, 3, 1, 2)
            
            # Apply updates with alive mask
            x = x + ds * alive_mask
            
            # Life dynamics
            neighbor_life = F.max_pool2d(alive_mask, kernel_size=3, stride=1, padding=1)
            life_mask = (neighbor_life > 0.001).float()
            life_mask = torch.maximum(life_mask, alive_mask * 0.7)
            x = x * life_mask
        
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0),
        )
        
        self._init_weights()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, img):
        return self.model(img).view(img.shape[0], -1)

class CrossEvaluator(nn.Module):
    """Evaluates both generator and NCA outputs"""
    def __init__(self, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, img):
        return self.model(img)

# ==================== DATASET ====================
class ImageDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.images = [f for f in os.listdir(root_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(self.images)} images in {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

# ==================== CHECKPOINT MANAGEMENT ====================
class CheckpointManager:
    """Robust checkpoint management with metrics tracking"""
    
    def __init__(self, checkpoint_dir, keep_last_n=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.keep_last_n = keep_last_n
        self.metrics_history = []
    
    def save_checkpoint(self, epoch, models, optimizers, scores=None):
        """Save checkpoint with metrics"""
        try:
            checkpoint_data = {
                'epoch': epoch,
                'models': {name: model.state_dict() for name, model in models.items()},
                'optimizers': {name: opt.state_dict() for name, opt in optimizers.items()},
                'metrics_history': self.metrics_history,
                'timestamp': time.time()
            }
            
            if scores:
                checkpoint_data['scores'] = scores
                self.metrics_history.append(scores)
            
            # Save main checkpoint
            latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
            torch.save(checkpoint_data, latest_path)
            
            # Save epoch checkpoint
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint_data, epoch_path)
            
            print(f"✅ Checkpoint saved: epoch {epoch}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, models, optimizers):
        """Load checkpoint and restore metrics"""
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        
        if not latest_path.exists():
            print("No checkpoint found, starting from scratch")
            return 0
        
        try:
            checkpoint = torch.load(latest_path, weights_only=False, map_location=Config.DEVICE)
            
            # Load model states
            for name, model in models.items():
                if name in checkpoint['models']:
                    model.load_state_dict(checkpoint['models'][name])
                    print(f"✅ Loaded {name} state")
            
            # Load optimizer states
            for name, optimizer in optimizers.items():
                if name in checkpoint['optimizers']:
                    optimizer.load_state_dict(checkpoint['optimizers'][name])
                    print(f"✅ Loaded {name} optimizer")
            
            # Restore metrics history
            if 'metrics_history' in checkpoint:
                self.metrics_history = checkpoint['metrics_history']
                print(f"✅ Loaded {len(self.metrics_history)} epochs of metrics")
            
            epoch = checkpoint['epoch']
            print(f"✅ Resumed from epoch {epoch}")
            return epoch
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return 0
    
    def _cleanup_old_checkpoints(self):
        """Remove old epoch checkpoints"""
        try:
            epoch_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if len(epoch_files) > self.keep_last_n:
                epoch_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in epoch_files[:-self.keep_last_n]:
                    old_file.unlink()
                    print(f"🗑️  Removed old checkpoint: {old_file.name}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

# ==================== TRAINING UTILITIES ====================
def tensor_to_b64(tensor):
    """Convert tensor to base64 for web display"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def update_status(status_message, images=None, scores=None, error=False):
    """Update training status for web interface"""
    status_data = {
        'message': status_message,
        'timestamp': time.time(),
        'error': error
    }
    
    if images:
        status_data['images'] = {k: tensor_to_b64(v) for k, v in images.items()}
    
    if scores:
        status_data['scores'] = scores
    
    # Save to status file
    try:
        with open('training_status.json', 'w') as f:
            json.dump(status_data, f)
    except Exception as e:
        print(f"Warning: Could not save status: {e}")

# ==================== MAIN TRAINING LOOP ====================
def consolidated_training_loop():
    """Main consolidated training loop with all features"""
    
    # Apply CPU optimizations
    cpu_workers = Config.apply_cpu_optimizations()
    
    # Setup graceful interruption
    interrupted = False
    
    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n🛑 Graceful shutdown initiated (signal {signum})")
        print("Finishing current batch and saving checkpoint...")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🚀 Starting Consolidated Training")
        print("=" * 60)
        print(f"Device: {Config.DEVICE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LR}")
        print(f"Enhanced Features: {ENHANCED_FEATURES}")
        print(f"W-Space Stabilizer: {W_STABILIZER_AVAILABLE}")
        print("=" * 60)
        
        # Initialize models
        generator = IntegratedGenerator(Config.Z_DIM, Config.W_DIM).to(Config.DEVICE)
        discriminator = Discriminator(Config.IMG_SIZE).to(Config.DEVICE)
        nca = IntegratedNCA(Config.NCA_CHANNELS, Config.W_DIM).to(Config.DEVICE)
        gen_evaluator = CrossEvaluator(Config.IMG_SIZE).to(Config.DEVICE)
        nca_evaluator = CrossEvaluator(Config.IMG_SIZE).to(Config.DEVICE)
        
        # Initialize W-space stabilizer if available
        w_stabilizer = None
        if W_STABILIZER_AVAILABLE:
            w_stabilizer = WSpaceStabilizer(Config.W_DIM).to(Config.DEVICE)
            print("✅ W-space stabilizer initialized")
        
        # Initialize enhanced systems if available
        cross_learning_system = None
        if ENHANCED_FEATURES:
            cross_learning_system = EnhancedCrossLearningSystem(Config.IMG_SIZE).to(Config.DEVICE)
            print("✅ Enhanced cross-learning system initialized")
        
        # Initialize optimizers
        gen_optimizer = optim.Adam(generator.parameters(), lr=Config.LR, betas=(0.5, 0.999))
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=Config.LR, betas=(0.5, 0.999))
        nca_optimizer = optim.Adam(nca.parameters(), lr=Config.LR, betas=(0.5, 0.999))
        gen_eval_optimizer = optim.Adam(gen_evaluator.parameters(), lr=Config.LR, betas=(0.5, 0.999))
        nca_eval_optimizer = optim.Adam(nca_evaluator.parameters(), lr=Config.LR, betas=(0.5, 0.999))
        
        optimizers = {
            'generator': gen_optimizer,
            'discriminator': disc_optimizer,
            'nca': nca_optimizer,
            'gen_evaluator': gen_eval_optimizer,
            'nca_evaluator': nca_eval_optimizer
        }
        
        if w_stabilizer:
            w_stab_optimizer = optim.Adam(w_stabilizer.parameters(), lr=Config.LR * 0.1)
            optimizers['w_stabilizer'] = w_stab_optimizer
        
        if cross_learning_system:
            cross_optimizer = optim.Adam(cross_learning_system.parameters(), lr=Config.LR * 0.5)
            optimizers['cross_learning'] = cross_optimizer
        
        # Initialize dataset
        dataset = ImageDataset(Config.DATA_DIR, Config.IMG_SIZE)
        dataloader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=cpu_workers,
            persistent_workers=True if cpu_workers > 0 else False,
            pin_memory=False,
            prefetch_factor=2 if cpu_workers > 0 else 2
        )
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(Config.CHECKPOINT_DIR, Config.KEEP_CHECKPOINTS)
        
        # Load checkpoint if exists
        models = {
            'generator': generator,
            'discriminator': discriminator,
            'nca': nca,
            'gen_evaluator': gen_evaluator,
            'nca_evaluator': nca_evaluator
        }
        
        if w_stabilizer:
            models['w_stabilizer'] = w_stabilizer
        if cross_learning_system:
            models['cross_learning'] = cross_learning_system
        
        start_epoch = checkpoint_manager.load_checkpoint(models, optimizers) + 1
        
        # Training loop
        print(f"🚀 Training started from epoch {start_epoch} - Press Ctrl+C to stop gracefully")
        
        for epoch in range(start_epoch, Config.EPOCHS + 1):
            if interrupted:
                print(f"🛑 Training interrupted at epoch {epoch}")
                break
            
            epoch_start_time = time.time()
            epoch_losses = {'gen': 0, 'disc': 0, 'nca': 0, 'gen_eval': 0, 'nca_eval': 0}
            
            for batch_idx, real_imgs in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
                if interrupted:
                    break
                
                real_imgs = real_imgs.to(Config.DEVICE)
                batch_size = real_imgs.shape[0]
                
                # Generate noise
                z = torch.randn(batch_size, Config.Z_DIM, device=Config.DEVICE)
                mixing_z = torch.randn(batch_size, Config.Z_DIM, device=Config.DEVICE)
                
                # Generate images
                fake_imgs, w = generator(z, return_w=True, mixing_noise=mixing_z)
                
                # Apply W-space stabilization if available
                if w_stabilizer:
                    w_variants, w_stats = w_stabilizer(w)
                    w = w_variants  # Use stabilized W
                
                # Generate NCA images
                nca_steps = random.randint(Config.NCA_STEPS_MIN, Config.NCA_STEPS_MAX)
                nca_seed = nca.get_seed(batch_size, Config.IMG_SIZE, Config.DEVICE)
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
                
                # Train Cross Evaluators
                gen_eval_optimizer.zero_grad()
                gen_eval_loss = F.mse_loss(gen_evaluator(fake_imgs), gen_evaluator(real_imgs))
                gen_eval_loss.backward()
                gen_eval_optimizer.step()
                
                nca_eval_optimizer.zero_grad()
                nca_eval_loss = F.mse_loss(nca_evaluator(nca_imgs), nca_evaluator(real_imgs))
                nca_eval_loss.backward()
                nca_eval_optimizer.step()
                
                # Enhanced cross-learning if available
                cross_learning_loss = 0
                if cross_learning_system:
                    cross_optimizer.zero_grad()
                    images_dict = {
                        'generator': fake_imgs,
                        'nca': nca_imgs,
                        'real': real_imgs
                    }
                    cross_learning_loss = cross_learning_system(
                        images_dict, 
                        epoch / Config.EPOCHS, 
                        batch_idx / len(dataloader)
                    )
                    cross_learning_loss.backward()
                    cross_optimizer.step()
                
                # Accumulate losses
                epoch_losses['gen'] += gen_loss.item()
                epoch_losses['disc'] += disc_loss.item()
                epoch_losses['nca'] += nca_loss.item()
                epoch_losses['gen_eval'] += gen_eval_loss.item()
                epoch_losses['nca_eval'] += nca_eval_loss.item()
                
                # Update status every 10 batches
                if batch_idx % 10 == 0:
                    status_msg = f"Epoch {epoch}/{Config.EPOCHS}, Batch {batch_idx}/{len(dataloader)}"
                    images = {
                        'real': real_imgs,
                        'generator': fake_imgs,
                        'nca': nca_imgs
                    }
                    update_status(status_msg, images=images)
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            
            # Calculate average losses
            for key in epoch_losses:
                epoch_losses[key] /= len(dataloader)
            
            # Create scores for checkpoint
            scores = {
                'gen_loss': epoch_losses['gen'],
                'disc_loss': epoch_losses['disc'],
                'nca_loss': epoch_losses['nca'],
                'gen_eval_loss': epoch_losses['gen_eval'],
                'nca_eval_loss': epoch_losses['nca_eval'],
                'cross_learning_loss': cross_learning_loss if isinstance(cross_learning_loss, (int, float)) else 0,
                'epoch_time': epoch_time
            }
            
            # Print progress
            print(f"\nEpoch {epoch}/{Config.EPOCHS} completed in {epoch_time:.2f}s")
            print(f"Gen: {scores['gen_loss']:.4f}, Disc: {scores['disc_loss']:.4f}, NCA: {scores['nca_loss']:.4f}")
            print(f"Gen Eval: {scores['gen_eval_loss']:.4f}, NCA Eval: {scores['nca_eval_loss']:.4f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_manager.save_checkpoint(epoch, models, optimizers, scores)
            
            # Update final status
            status_msg = f"Completed epoch {epoch}/{Config.EPOCHS}"
            images = {
                'real': real_imgs,
                'generator': fake_imgs,
                'nca': nca_imgs
            }
            update_status(status_msg, images=images, scores=scores)
        
        # Final checkpoint save
        checkpoint_manager.save_checkpoint(epoch, models, optimizers, scores)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        update_status(error_msg, error=True)
        return False
    
    return True

# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(Config.SAMPLES_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Run training
    success = consolidated_training_loop()
    
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("💥 Training failed!")
        sys.exit(1) 