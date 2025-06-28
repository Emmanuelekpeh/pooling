import os
import io
import json
import time
import base64
import random
import argparse
from threading import Thread
import math

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
from enhanced_cross_learning_fixed import SignalWeightingNetwork, EnhancedCrossLearningSystem
import traceback
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam

# --- Configuration ---
Z_DIM = 64  # Reduced for memory
W_DIM = 128  # Increased for richer style representation
IMG_SIZE = 64
NCA_CHANNELS = 8  # Reduced for memory
BATCH_SIZE = 4  # Slightly increased for better training dynamics
LR = 2e-4   # Adjusted learning rate
EPOCHS = 1550
NCA_STEPS_MIN = 18  # Increased for better growth - needs more steps to fill 64x64 from 5x5 seed
NCA_STEPS_MAX = 38  # Increased for better growth - allows full image coverage
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/ukiyo-e-hybrid"  # Use NEW 70/30 hybrid dataset (28 traditional + 12 pixel art from 225 images)
SAMPLES_DIR = "./samples"
# Use environment variable for checkpoint directory with fallback to relative path
# This ensures it works with the Fly.io mounted volume at /app/checkpoints
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints" if os.path.exists("/app") else "./checkpoints")
KEEP_CHECKPOINTS = 5  # Number of recent epoch checkpoints to keep

# --- Models ---
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.pixel_norm = PixelNorm()
        
        # 8-layer mapping network with equalized learning rate
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
        
        # Initialize weights
        self.truncation_psi = 0.7
        self.truncation_cutoff = 8
        self.register_buffer('w_avg', torch.zeros(w_dim))
        
    def forward(self, z, truncation=True):
        x = self.pixel_norm(z)
        w = self.mapping(x)
        
        # Update moving average of w during training
        if self.training:
            self.w_avg.copy_(w.detach().mean(0).lerp(self.w_avg, 0.995))
        
        # Apply truncation trick
        if truncation:
            w = self.w_avg.lerp(w, self.truncation_psi)
            
        return w

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (2 / in_dim) ** 0.5
        
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

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

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # AdaIN layers with gradient scaling
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)
        
        # Initialize noise weights with small random values instead of zeros
        self.noise_weight1 = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.01)
        self.noise_weight2 = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.01)
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, w):
        # First conv + noise + activation + AdaIN with gradient control
        x = self.conv1(x)
        noise1 = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_weight1 * noise1
        x = self.activation(x)
        x = self.adain1(x, w)
        x = torch.clamp(x, -10.0, 10.0)  # Prevent extreme values
        
        # Second conv + noise + activation + AdaIN with gradient control
        x = self.conv2(x)
        noise2 = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_weight2 * noise2
        x = self.activation(x)
        x = self.adain2(x, w)
        x = torch.clamp(x, -10.0, 10.0)  # Prevent extreme values
        
        return x

class IntegratedGenerator(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim)
        
        # Initial learned constant input with better initialization
        self.const = nn.Parameter(torch.randn(1, 256, 4, 4) * 0.1)
        
        # Progressive growing blocks with adjusted channels
        self.blocks = nn.ModuleList([
            GeneratorBlock(256, 128, w_dim),  # 4x4 -> 8x8
            GeneratorBlock(128, 64, w_dim),   # 8x8 -> 16x16
            GeneratorBlock(64, 32, w_dim),    # 16x16 -> 32x32
            GeneratorBlock(32, 16, w_dim),    # 32x32 -> 64x64
        ])
        
        # To RGB layers with better initialization
        self.to_rgb = nn.ModuleList([
            EqualizedConv2d(128, 3, 1),
            EqualizedConv2d(64, 3, 1),
            EqualizedConv2d(32, 3, 1),
            EqualizedConv2d(16, 3, 1),
        ])
        
        # Initialize alpha for smooth growing
        self.register_buffer('alpha', torch.tensor(1.0))
        self.current_block = len(self.blocks) - 1  # Start at full resolution
        
        # Initialize weights with smaller values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better stability"""
        for m in self.modules():
            if isinstance(m, EqualizedConv2d):
                # Use smaller initialization for conv layers
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def set_alpha(self, alpha):
        """Set the alpha value for progressive growing transition."""
        self.alpha.fill_(alpha)
    
    def style_mixing(self, w1, w2, mixing_prob=0.9):
        """Apply style mixing regularization."""
        if self.training and torch.rand(1) < mixing_prob:
            # Randomly choose crossover point
            crossover = torch.randint(1, len(self.blocks), (1,)).item()
            w_mixed = []
            for i in range(len(self.blocks)):
                w_mixed.append(w1 if i < crossover else w2)
            return w_mixed
        return [w1] * len(self.blocks)
    
    def forward(self, noise, return_w=False, mixing_noise=None):
        # Map noise to W space with gradient control
        w = self.mapping(noise)
        w = torch.clamp(w, -10.0, 10.0)  # Prevent extreme style values
        
        # Style mixing
        if mixing_noise is not None and self.training:
            w2 = self.mapping(mixing_noise)
            w2 = torch.clamp(w2, -10.0, 10.0)
            w = self.style_mixing(w, w2)
        else:
            w = [w] * len(self.blocks)
        
        # Start from learned constant
        x = self.const.repeat(noise.shape[0], 1, 1, 1)
        
        # Progressive generation through blocks with gradient control
        rgb = None
        rgb_prev = None
        
        for i, (block, to_rgb) in enumerate(zip(self.blocks, self.to_rgb)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x, w[i])
            
            # Handle progressive growing
            if i == self.current_block - 1:
                # Previous resolution RGB
                rgb_prev = to_rgb(x)
                rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='bilinear', align_corners=False)
            elif i == self.current_block:
                # Current resolution RGB
                rgb = to_rgb(x)
                # Fade in new layer during transition with gradient control
                if self.alpha < 1 and rgb_prev is not None:
                    rgb = self.alpha * rgb + (1 - self.alpha) * rgb_prev
        
        # Apply tanh to final output with gradient control
        rgb = torch.tanh(rgb)
        rgb = torch.clamp(rgb, -0.99, 0.99)  # Prevent extreme values at tanh boundaries
        
        if return_w:
            return rgb, w[0]  # Return first w for compatibility
        return rgb

class SimpleGrowthNCA(nn.Module):
    """Breakthrough SimpleGrowthNCA - Clean, simple NCA focused purely on growth"""
    def __init__(self, n_channels=8, w_dim=128):
        super().__init__()
        self.n_channels = n_channels
        self.w_dim = w_dim
        
        # Simple perception - using breakthrough Sobel filters
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Simple update network - exactly like breakthrough
        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 3, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, n_channels, 1)
        )
        
        # Growth parameters - key breakthrough parameters
        self.growth_rate = nn.Parameter(torch.tensor(0.3))
        self.alive_threshold = 0.1
        
        # Style conditioning layers (minimal for integration)
        self.style_mod = nn.Linear(w_dim, n_channels)
        
        # Initialize for growth - exactly like breakthrough
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.05)  # Slight positive bias
    
    def perceive(self, x):
        """Simple perception - exactly like breakthrough"""
        sobel_x_expanded = self.sobel_x.repeat(self.n_channels, 1, 1, 1)
        sobel_y_expanded = self.sobel_y.repeat(self.n_channels, 1, 1, 1)
        
        dx = F.conv2d(x, sobel_x_expanded, groups=self.n_channels, padding=1)
        dy = F.conv2d(x, sobel_y_expanded, groups=self.n_channels, padding=1)
        
        return torch.cat([x, dx, dy], dim=1)
    
    def get_seed(self, batch_size, size, device, seed_type="distributed"):
        """Create initial seed - using breakthrough parameters"""
        seed = torch.zeros(batch_size, self.n_channels, size, size, device=device)
        
        if seed_type == "center":
            center = size // 2
            radius = 3
            # Create circular seed
            y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
            dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
            mask = dist <= radius
            
            for b in range(batch_size):
                # Set RGB channels with some color - stronger initial values like breakthrough
                seed[b, 0, mask] = 0.5  # Red
                seed[b, 1, mask] = 0.3  # Green  
                seed[b, 2, mask] = 0.7  # Blue
                seed[b, 3, mask] = 1.0  # Alpha (alive) - strong like breakthrough
                
                # Initialize hidden channels
                for i in range(4, self.n_channels):
                    seed[b, i, mask] = torch.randn(mask.sum(), device=device) * 0.1
        
        elif seed_type == "distributed":
            # Distributed seeding for batch training
            for b in range(batch_size):
                num_seeds = torch.randint(1, 3, (1,)).item()  # 1-2 seeds
                margin = size // 6
                
                for _ in range(num_seeds):
                    center_x = torch.randint(margin, size - margin, (1,)).item()
                    center_y = torch.randint(margin, size - margin, (1,)).item()
                    radius = torch.randint(2, 4, (1,)).item()
                    
                    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    mask = dist <= radius
                    
                    # Set seed values
                    seed[b, 0, mask] = torch.rand(1).item() * 0.8 - 0.4  # RGB
                    seed[b, 1, mask] = torch.rand(1).item() * 0.8 - 0.4
                    seed[b, 2, mask] = torch.rand(1).item() * 0.8 - 0.4
                    seed[b, 3, mask] = 0.8  # Strong alpha
                    
                    # Hidden channels
                    for i in range(4, self.n_channels):
                        seed[b, i, mask] = torch.randn(mask.sum(), device=device) * 0.1
        
        return seed

    def to_rgb(self, x):
        """Convert NCA state to RGB image"""
        rgb = x[:, :3, :, :]
        rgb = torch.tanh(rgb)  # Ensure values are in [-1, 1]
        return rgb

    def to_rgba(self, x):
        """Convert NCA state to RGBA image"""
        rgb = x[:, :3, :, :]
        rgb = torch.tanh(rgb)  # Ensure RGB values are in [-1, 1]
        
        alpha = x[:, 3:4, :, :]
        alpha = alpha.clamp(0, 1)  # Ensure alpha is in [0, 1]
        
        return torch.cat([rgb, alpha], dim=1)
    
    def forward(self, x, w, steps, growth_boost=1.0, target_img=None):
        """Clean forward pass - breakthrough approach with minimal style conditioning"""
        # Optional style modulation (minimal impact)
        style_bias = self.style_mod(w).view(w.shape[0], self.n_channels, 1, 1) * 0.1  # Very weak conditioning
        
        for step in range(steps):
            # Perception
            perceived = self.perceive(x)
            
            # Update
            dx = self.update_net(perceived)
            
            # Apply growth rate with boost - THE KEY BREAKTHROUGH
            dx = dx * self.growth_rate * growth_boost
            
            # Add minimal style conditioning
            dx = dx + style_bias
            
            # Growth encouragement for low populations - KEY BREAKTHROUGH MECHANISM
            alive_mask = (x[:, 3:4] > self.alive_threshold).float()
            alive_ratio = alive_mask.mean()
            
            if alive_ratio < 0.2:
                # Emergency growth boost - exactly like breakthrough
                neighbor_alive = F.avg_pool2d(alive_mask, 3, stride=1, padding=1)
                growth_zones = (neighbor_alive > 0.01).float()
                
                # Create growth boost tensor
                growth_boost_tensor = torch.zeros_like(dx)
                growth_boost_tensor[:, 3:4] = growth_zones * 0.2  # Alpha boost
                
                dx = dx + growth_boost_tensor
            
            # Update state
            x = x + dx
            
            # Simple bounds - avoid in-place operations
            x = torch.clamp(x, -1, 1)
            alpha_clamped = torch.clamp(x[:, 3:4], 0, 1)  # Alpha in [0,1]
            x = torch.cat([x[:, :3], alpha_clamped, x[:, 4:]], dim=1)
        
        return x

def get_progressive_growth_boost(epoch, total_epochs=500):
    """Progressive growth curriculum - THE KEY BREAKTHROUGH INSIGHT"""
    # Adapted for main training (500 epochs)
    if epoch < total_epochs * 0.2:  # First 20% (100 epochs)
        return 2.0  # Strong early growth
    elif epoch < total_epochs * 0.4:  # Next 20% (200 epochs)
        return 1.5  # Moderate growth
    elif epoch < total_epochs * 0.6:  # Next 20% (300 epochs)
        return 1.2  # Light boost
    else:  # Final 40% (300-500 epochs)
        return 1.0  # Normal growth

# Replace the complex IntegratedNCA class
IntegratedNCA = SimpleGrowthNCA

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.model = nn.Sequential(
            # Initial convolution
            EqualizedConv2d(3, 16, 3, 2, 1),    # 64 -> 32
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            
            # Downsampling blocks
            EqualizedConv2d(16, 32, 3, 2, 1),   # 32 -> 16
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            
            EqualizedConv2d(32, 64, 3, 2, 1),   # 16 -> 8
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            EqualizedConv2d(64, 128, 3, 2, 1),  # 8 -> 4
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            
            # Global average pooling and final convolution
            nn.AdaptiveAvgPool2d(1),            # 4 -> 1
            nn.Flatten(),                        # Flatten to vector
            nn.Linear(128, 1)                    # Final prediction
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        return self.model(img)

# --- Cross-Evaluation Networks ---
class CrossEvaluator(nn.Module):
    """Network that evaluates the quality of images from the other model"""
    def __init__(self, img_size):
        super().__init__()
        self.evaluator = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),  # 64 -> 32, reduced from 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), # 32 -> 16, reduced from 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), # 16 -> 8, reduced from 128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # 8 -> 4, reduced from 256
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),  # Reduced from 128
            nn.ReLU(),
            nn.Linear(64, 1),    # Quality score 0-1, reduced from 128
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.evaluator(img).squeeze()

class NCAEvaluator(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.evaluator = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(3, 16, 3, 2, 1),      # 64 -> 32
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, 3, 2, 1),     # 32 -> 16
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3, 2, 1),     # 16 -> 8
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, 2, 1),    # 8 -> 4
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            
            # Global pooling and final prediction
            nn.AdaptiveAvgPool2d(1),        # Global average pooling
            nn.Flatten(),
            nn.Linear(128, 1),              # Final quality score
            nn.Sigmoid()                     # Normalize to [0,1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        return self.evaluator(img).squeeze()

# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
        print(f"Found {len(self.image_files)} images in {root_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img_path = os.path.normpath(img_path)
        try:
            image = Image.open(img_path).convert('RGB')
            transformed = self.transform(image)
            return transformed
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image on error
            return torch.zeros(3, self.img_size, self.img_size)

# --- Checkpoint Management ---
def save_checkpoint(epoch, generator, discriminator, gen_evaluator, nca_evaluator,
                    transformer_critic, gen_optimizer, disc_optimizer):
    """Save a checkpoint of the training state."""
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'gen_evaluator_state_dict': gen_evaluator.state_dict(),
            'nca_evaluator_state_dict': nca_evaluator.state_dict(),
            'transformer_critic_state_dict': transformer_critic.state_dict(),
            'gen_optimizer_state_dict': gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': disc_optimizer.state_dict(),
            'scores': update_status.metrics_history if hasattr(update_status, 'metrics_history') else []
        }
        
        # Save the checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Update latest checkpoint symlink
        latest_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        torch.save(checkpoint, latest_path)
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(CHECKPOINT_DIR, KEEP_CHECKPOINTS)
        
        print(f"Saved checkpoint at epoch {epoch}")
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        traceback.print_exc()

def load_checkpoint(checkpoint_path, generator, discriminator, gen_evaluator, nca_evaluator,
                   transformer_critic, gen_optimizer, disc_optimizer):
    """Load a training checkpoint."""
    try:
        if not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch")
            return 1
        
        checkpoint = torch.load(checkpoint_path)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gen_evaluator.load_state_dict(checkpoint['gen_evaluator_state_dict'])
        nca_evaluator.load_state_dict(checkpoint['nca_evaluator_state_dict'])
        transformer_critic.load_state_dict(checkpoint['transformer_critic_state_dict'])
        
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        # Restore metrics history
        if 'scores' in checkpoint:
            update_status.metrics_history = checkpoint['scores']
            print(f"Restored {len(update_status.metrics_history)} epochs of metrics history")
        
        epoch = checkpoint.get('epoch', 1)
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        traceback.print_exc()
        return 1

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5):
    """Remove old epoch checkpoints, keeping only the last N"""
    try:
        # Get all epoch checkpoint files
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                # Extract epoch number
                try:
                    epoch_num = int(filename.replace('checkpoint_epoch_', '').replace('.pt', ''))
                    filepath = os.path.join(checkpoint_dir, filename)
                    checkpoint_files.append((epoch_num, filepath))
                except ValueError:
                    continue
        
        # Sort by epoch number (newest first)
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints beyond keep_last_n
        if len(checkpoint_files) > keep_last_n:
            files_to_remove = checkpoint_files[keep_last_n:]
            for epoch_num, filepath in files_to_remove:
                try:
                    os.remove(filepath)
                    print(f"Removed old checkpoint: {os.path.basename(filepath)}")
                except Exception as e:
                    print(f"Error removing {filepath}: {str(e)}")
            
            print(f"Cleanup complete: Kept {min(len(checkpoint_files), keep_last_n)} most recent checkpoints")
        else:
            print(f"No cleanup needed: {len(checkpoint_files)} checkpoints (≤ {keep_last_n})")
            
    except Exception as e:
        print(f"Error during checkpoint cleanup: {str(e)}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NCA vs StyleGAN Training</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: sans-serif; background-color: #282c34; color: #abb2bf; text-align: center; }
        .main-container { max-width: 1800px; margin: 0 auto; padding: 20px; }
        .charts-container { display: flex; justify-content: space-around; margin: 40px 0; flex-wrap: wrap; gap: 20px; }
        .chart-box { background-color: #3b4048; border-radius: 10px; padding: 20px; width: 45%; min-width: 500px; }
        .container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
        @media (max-width: 1800px) { 
            .container { grid-template-columns: repeat(2, 1fr); }
            .chart-box { width: 100%; min-width: 300px; }
        }
        @media (max-width: 900px) { 
            .container { grid-template-columns: 1fr; }
            .charts-container { flex-direction: column; }
        }
        .column { display: flex; flex-direction: column; align-items: center; }
        h1 { color: #61afef; font-size: 2em; margin: 20px 0; }
        h2 { color: #61afef; font-size: 1.2em; margin: 10px 0; }
        h3 { color: #e5c07b; margin: 0 0 15px 0; font-size: 1.1em; }
        img { border: 2px solid #61afef; width: 100%; max-width: 400px; height: 400px; image-rendering: pixelated; object-fit: contain; }
        #status { margin-top: 20px; font-size: 1.2em; min-height: 50px; padding: 10px; border-radius: 5px; background-color: #3b4048; }
        .error { color: #e06c75; border: 1px solid #e06c75; }
        .scores { margin-top: 10px; font-size: 0.9em; color: #98c379; }
        #metrics { width: 100%; height: 400px; }
    </style>
</head>
<body>
    <div class="main-container">
        <h1>🎨 NCA vs StyleGAN Training Dashboard</h1>
        <div id="status">Connecting...</div>
        
        <!-- Training Progress Charts -->
        <div class="charts-container">
            <div class="chart-box">
                <h3>📊 Training Metrics</h3>
                <div id="metrics"></div>
            </div>
        </div>
        
        <!-- Training Images -->
        <div class="container">
        <div class="column">
            <h2>Real Target Images</h2>
            <img id="target-image" src="https://via.placeholder.com/400" alt="Target Images">
        </div>
        <div class="column">
            <h2>Generator Output</h2>
            <img id="generator-image" src="https://via.placeholder.com/400" alt="Generator Output">
            <div id="gen-scores" class="scores">Quality Score: --</div>
        </div>
        <div class="column">
            <h2>NCA Output</h2>
            <img id="nca-image" src="https://via.placeholder.com/400" alt="NCA Output">
            <div id="nca-scores" class="scores">Quality Score: --</div>
        </div>
        <div class="column">
            <h2>Transformer Output</h2>
            <img id="transformer-image" src="https://via.placeholder.com/400" alt="Transformer Output">
            <div id="transformer-scores" class="scores">Mode: --</div>
        </div>
    </div>
    </div>
    <script>
        function updateMetricsChart(metricsHistory) {
            const traces = {};
            
            // Initialize traces for each metric
            if (metricsHistory && metricsHistory.length > 0) {
                Object.keys(metricsHistory[0]).forEach(key => {
                    if (key !== 'timestamp' && key !== 'transformer_mode') {
                        traces[key] = {
                            x: [],
                            y: [],
                            name: key,
                            type: 'scatter',
                            mode: 'lines'
                        };
                    }
                });
                
                // Populate trace data
                metricsHistory.forEach(point => {
                    Object.keys(traces).forEach(key => {
                        traces[key].x.push(point.timestamp);
                        traces[key].y.push(point[key]);
                    });
                });
                
                const layout = {
                    title: 'Training Metrics',
                    plot_bgcolor: '#3b4048',
                    paper_bgcolor: '#282c34',
                    font: {
                        color: '#abb2bf'
                    },
                    xaxis: {
                        title: 'Time',
                        gridcolor: '#4b5363'
                    },
                    yaxis: {
                        title: 'Value',
                        gridcolor: '#4b5363'
                    },
                    showlegend: true,
                    legend: {
                        bgcolor: '#3b4048',
                        font: { color: '#abb2bf' }
                    }
                };
                
                Plotly.newPlot('metrics', Object.values(traces), layout);
            }
        }
        
        function pollStatus() {
            setInterval(() => {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        statusDiv.textContent = data.status || 'No status message.';
                        if (data.error) {
                            statusDiv.classList.add('error');
                        } else {
                            statusDiv.classList.remove('error');
                        }

                        if (data.images && data.images.length > 0) {
                            if (data.images[0]) document.getElementById('target-image').src = 'data:image/png;base64,' + data.images[0];
                            if (data.images[1]) document.getElementById('generator-image').src = 'data:image/png;base64,' + data.images[1];
                            if (data.images[2]) document.getElementById('nca-image').src = 'data:image/png;base64,' + data.images[2];
                            if (data.images[3] && data.images[3] !== null) {
                                document.getElementById('transformer-image').src = 'data:image/png;base64,' + data.images[3];
                            }
                        }
                        
                        if (data.scores) {
                            document.getElementById('gen-scores').textContent = `Quality Score: ${data.scores.gen_quality?.toFixed(3) || '--'}`;
                            document.getElementById('nca-scores').textContent = `Quality Score: ${data.scores.nca_quality?.toFixed(3) || '--'}`;
                            document.getElementById('transformer-scores').textContent = `Mode: ${data.scores.transformer_mode || '--'}`;
                        }
                        
                        if (data.metrics_history) {
                            updateMetricsChart(data.metrics_history);
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching status:', error);
                        document.getElementById('status').textContent = 'Error: Could not connect to the server.';
                        document.getElementById('status').classList.add('error');
                    });
            }, 3000);
        }
        document.addEventListener('DOMContentLoaded', pollStatus);
    </script>
</body>
</html>
"""

# --- File-based state for inter-process communication ---
STATUS_FILE = os.path.join(SAMPLES_DIR, 'status.json')

# --- Mini-graph utilities for progress bar ---
def create_mini_graph(values, width=8, height=3):
    """Create a small ASCII graph from a list of values"""
    if not values or len(values) < 2:
        return [" " * width for _ in range(height)]
    
    # Normalize values to fit in height
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        normalized = [height // 2] * len(values)
    else:
        normalized = [int((v - min_val) / (max_val - min_val) * (height - 1)) for v in values]
    
    # Take only the last `width` values
    if len(normalized) > width:
        normalized = normalized[-width:]
    
    # Create the graph
    graph = []
    for row in range(height - 1, -1, -1):  # Top to bottom
        line = ""
        for i in range(len(normalized)):
            if normalized[i] == row:
                line += "●"
            elif normalized[i] > row:
                line += "│"
            else:
                line += " "
        # Pad to width
        line += " " * (width - len(line))
        graph.append(line)
    
    return graph

def format_mini_graphs(loss_history, metric_names, width=8):
    """Format multiple mini-graphs for display in progress bar"""
    if not loss_history:
        return ""
    
    graphs = {}
    for metric in metric_names:
        values = [epoch.get(metric, 0) for epoch in loss_history if isinstance(epoch.get(metric, 0), (int, float))]
        graphs[metric] = create_mini_graph(values, width=width)
    
    # Format for single line display
    graph_lines = []
    for metric in metric_names:
        if metric in graphs:
            # Take middle line of the graph for compact display
            middle_line = graphs[metric][len(graphs[metric]) // 2]
            graph_lines.append(f"{metric[:3]}:{middle_line}")
    
    return " | ".join(graph_lines)

def tensor_to_b64(tensor):
    """Convert a PyTorch tensor to a base64 encoded image string."""
    try:
        # Handle None input
        if tensor is None:
            return None
            
        # Ensure tensor is on CPU and detached from graph
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach().cpu()
        
        # Handle different tensor shapes
        if len(tensor.shape) == 2:  # Single channel [H, W]
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # Convert to [3, H, W]
        elif len(tensor.shape) == 3:  # [C, H, W]
            if tensor.shape[0] == 1:  # Single channel
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] == 4:  # RGBA
                tensor = tensor[:3]  # Take only RGB channels
        elif len(tensor.shape) == 4:  # [B, C, H, W]
            tensor = tensor[0]  # Take first batch
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] == 4:
                tensor = tensor[:3]
        
        # Ensure values are in [0, 1] range
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Convert to PIL image and then to base64
        img = transforms.ToPILImage()(tensor)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Warning: {str(e)}")
        return None

def update_status(status_message, images=None, scores=None, error=False):
    """Update the training status with images and scores."""
    try:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        
        # Initialize metrics history if it doesn't exist
        if not hasattr(update_status, 'metrics_history'):
            update_status.metrics_history = []
        
        # Process images
        b64_images = []
        if images:
            if isinstance(images, dict):
                images = [img for img in images.values() if img is not None]
            
            for img_tensor in images:
                try:
                    b64_img = tensor_to_b64(img_tensor)
                    b64_images.append(b64_img)  # Append even if None - maintains index alignment
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    b64_images.append(None)  # Maintain index alignment on error
        
        # Format and save metrics
        if scores:
            metrics = {
                # === PROPER GAN DISCRIMINATOR BREAKDOWN ===
                'disc_real_loss': scores.get('disc_real_loss', 0),
                'disc_fake_gen_loss': scores.get('disc_fake_gen_loss', 0),
                'disc_fake_nca_loss': scores.get('disc_fake_nca_loss', 0),
                'disc_total_loss': scores.get('disc_total_loss', 0),
                
                # === ADVERSARIAL LOSSES ===
                'gen_adversarial_loss': scores.get('gen_adversarial_loss', 0),
                'nca_adversarial_loss': scores.get('nca_adversarial_loss', 0),
                
                # === LEGACY MAPPING (for existing charts) ===
                'gen_eval_loss': scores.get('disc_loss', 0),
                'nca_eval_loss': scores.get('nca_loss', 0),
                'gen_penalty': scores.get('gen_loss', 0),
                'nca_penalty': scores.get('nca_penalty', 0),
                'feature_match_gen': scores.get('gen_feature_loss', 0),
                'feature_match_nca': scores.get('nca_feature_loss', 0),
                'transformer_loss': scores.get('transformer_loss', 0),
                'gen_quality': scores.get('gen_quality', 0),
                'nca_quality': scores.get('nca_quality', 0),
                'ensemble_prediction': scores.get('ensemble_quality', 0),
                'cross_learning_loss': scores.get('cross_learning_loss', 0)
            }
            update_status.metrics_history.append(metrics)
        
        # Save current status
        status = {
            'status': status_message,
            'images': b64_images,
            'scores': scores,
            'error': error
        }
        
        with open(os.path.join(SAMPLES_DIR, 'status.json'), 'w') as f:
            json.dump(status, f)
            
    except Exception as e:
        print(f"Error in update_status: {str(e)}")
        traceback.print_exc()

# --- Training Function (to be run by the 'worker' process) ---
def training_loop():
    """The main training loop with progressive growing and style mixing."""
    try:
        # Apply CPU optimizations first
        print("🖥️  Applying CPU optimizations...")
        num_threads = os.cpu_count()
        torch.set_num_threads(num_threads)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        torch.set_float32_matmul_precision('high')
        print(f"✅ PyTorch threads: {num_threads}")
        
        # Setup graceful interruption handling
        import signal
        interrupted = False
        
        def signal_handler(signum, frame):
            nonlocal interrupted
            print("\n🛑 Graceful interruption requested... finishing current batch")
            interrupted = True
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initialize models
        generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
        discriminator = Discriminator(IMG_SIZE).to(DEVICE)
        gen_evaluator = CrossEvaluator(IMG_SIZE).to(DEVICE)
        nca_evaluator = NCAEvaluator(img_size=IMG_SIZE).to(DEVICE)
        transformer_critic = TransformerCritic(img_size=IMG_SIZE, dim=256, depth=6, heads=8).to(DEVICE)
        
        # Initialize NCA model with breakthrough parameters
        nca_model = SimpleGrowthNCA(n_channels=8, w_dim=W_DIM).to(DEVICE)  # 8 channels like breakthrough
        
        # Initialize optimizers with DISCRIMINATOR COLLAPSE FIXES (Neptune.ai research)
        # Fix: Discriminator needs higher LR, generators need lower LR to prevent collapse
        gen_optimizer = optim.Adam(
            generator.parameters(),
            lr=LR * 0.1,   # Further reduced from 0.25 (now 0.00002) - quarter generator LR for stability
            betas=(0.0, 0.99),
            eps=1e-8,
            weight_decay=1e-5
        )
        disc_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=LR * 1.5,   # Further increased from 1.0 (now 0.0003) - boost discriminator LR for better feedback
            betas=(0.0, 0.99),
            eps=1e-8,
            weight_decay=1e-5
        )
        transformer_optimizer = optim.Adam(
            transformer_critic.parameters(),
            lr=LR * 0.1,  # Lower learning rate for transformer
            betas=(0.0, 0.99),
            eps=1e-8,
            weight_decay=1e-5
        )
        nca_optimizer = optim.Adam(  # Add NCA optimizer with discriminator collapse fix
            nca_model.parameters(),
            lr=LR * 0.1,   # Further reduced from 0.25 (now 0.00002) - quarter NCA LR for stability
            betas=(0.0, 0.99),
            eps=1e-8,
            weight_decay=1e-5
        )
        
        # Load checkpoint if exists - with breakthrough compatibility check
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
        if os.path.exists(latest_checkpoint):
            try:
            checkpoint = torch.load(latest_checkpoint, weights_only=False, map_location=DEVICE)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            gen_evaluator.load_state_dict(checkpoint['gen_evaluator_state_dict'])
            nca_evaluator.load_state_dict(checkpoint['nca_evaluator_state_dict'])
            transformer_critic.load_state_dict(checkpoint['transformer_critic_state_dict'])
            gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer_state_dict'])
                
                # Try to load NCA optimizer, but skip if incompatible (breakthrough model change)
                try:
            nca_optimizer.load_state_dict(checkpoint['nca_optimizer_state_dict'])
                except (KeyError, RuntimeError) as e:
                    print(f"⚠️  NCA optimizer incompatible (breakthrough model change) - starting fresh: {e}")
                    # Reset NCA optimizer for breakthrough model
                    nca_optimizer = optim.Adam(
                        nca_model.parameters(),
                        lr=LR * 0.5,
                        betas=(0.0, 0.99),
                        eps=1e-8,
                        weight_decay=1e-5
                    )
            
            # Restore metrics history for charts
            if 'scores' in checkpoint and checkpoint['scores']:
                update_status.metrics_history = checkpoint['scores']
                print(f"📊 Restored {len(checkpoint['scores'])} epochs of metrics history")
            else:
                update_status.metrics_history = []
                print("📊 No metrics history found in checkpoint")
            
            start_epoch = checkpoint['epoch'] + 1
                print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} (with breakthrough NCA)")
                
            except Exception as e:
                print(f"⚠️  Checkpoint loading failed (breakthrough model change) - starting fresh: {e}")
                start_epoch = 1
                update_status.metrics_history = []
        else:
            start_epoch = 1
            update_status.metrics_history = []
            print("🆕 No checkpoint found, starting from scratch with breakthrough NCA")
        
        # Initialize dataset with CPU-optimized settings
        dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)  # Use pixel art dataset
        cpu_workers = min(4, num_threads // 2)  # Don't overwhelm CPU
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=cpu_workers,
            persistent_workers=True if cpu_workers > 0 else False,
            pin_memory=False,  # Not needed for CPU
            prefetch_factor=2 if cpu_workers > 0 else 2
        )
        print(f"✅ DataLoader workers: {cpu_workers}")
        
        print("Training configuration:")
        print(f"- IMG_SIZE: {IMG_SIZE}")
        print(f"- BATCH_SIZE: {BATCH_SIZE}")
        print(f"- DEVICE: {DEVICE}")
        print(f"- Dataset size: {len(dataset)}")
        print(f"- Starting from epoch: {start_epoch}")
        
        # Training loop with graceful interruption
        print("🚀 Training started - Press Ctrl+C to stop gracefully")
        print("🌱 Using Progressive Growth Curriculum (breakthrough approach)")
        for epoch in range(start_epoch, EPOCHS + 1):
            if interrupted:
                print(f"🛑 Training interrupted at epoch {epoch}")
                break
                
            # Get progressive growth boost - THE KEY BREAKTHROUGH
            growth_boost = get_progressive_growth_boost(epoch, EPOCHS)
                
            # === CYCLICAL TRANSFORMER MODE SWITCHING ===
            transformer_info = get_cyclical_transformer_mode(epoch)
            current_mode = transformer_info['mode']
            is_isolated = transformer_info['isolated']
            phase_name = transformer_info['phase_name']
            cycle_num = transformer_info['cycle_number']
            epochs_in_phase = transformer_info['epochs_in_phase']
            phase_progress = transformer_info['phase_progress']
            
            # Set transformer mode
            if is_isolated:
                transformer_critic.set_mode(current_mode, isolation_epochs=epochs_in_phase)
            else:
                transformer_critic.set_mode(current_mode)
            
            # Print phase transitions and adjust learning rates
            if epochs_in_phase == 1:  # First epoch of a new phase
                print(f"\n🔄 EPOCH {epoch}: TRANSFORMER CYCLE {cycle_num} - Entering {phase_name} phase")
                if phase_name == "GENERATION":
                    print(f"🎨 Extended generation phase: 120 epochs for experimentation!")
                    # Boost learning rate for active generation experimentation
                    for param_group in transformer_optimizer.param_groups:
                        param_group['lr'] *= 2.0  # Double LR for generation experiments
                elif phase_name == "ISOLATION":
                    print(f"🧠 Consolidation phase: Processing learned patterns")
                    # Reduce learning rate for careful consolidation
                    for param_group in transformer_optimizer.param_groups:
                        param_group['lr'] *= 0.1  # 10x reduction for isolation
                elif phase_name == "CRITIC":
                    print(f"🎯 Evaluation phase: Learning quality assessment")
                    # Reset to base learning rate for criticism
                    for param_group in transformer_optimizer.param_groups:
                        param_group['lr'] = LR * 0.5  # Reset to base transformer LR
            
            for batch_idx, real_imgs in enumerate(dataloader):
                if interrupted:
                    print(f"🛑 Stopping at batch {batch_idx} (safe interruption)")
                    break
                real_imgs = real_imgs.to(DEVICE)
                batch_size = real_imgs.size(0)
                
                # DISCRIMINATOR TRAINING FREQUENCY FIX (Neptune.ai research)
                # Train discriminator more often than generators to prevent collapse
                disc_steps = 2 if epoch <= 50 else 1  # Extra discriminator training in early epochs
                
                for disc_step in range(disc_steps):
                # Train Discriminator
                disc_optimizer.zero_grad()
                
                    # Real images with NOISE INJECTION (Neptune.ai fix for discriminator collapse)
                    # Add small noise to prevent discriminator from being too confident
                    noise_real = torch.randn_like(real_imgs) * 0.05  # 5% noise injection
                    disc_real = discriminator(real_imgs + noise_real)
                    # LABEL SMOOTHING: Use 0.9 instead of 1.0 for real labels (Neptune.ai recommendation)
                disc_real_loss = F.binary_cross_entropy_with_logits(
                        disc_real, torch.ones_like(disc_real) * 0.9  # Label smoothing for real images
                )
                
                # Fake images from Generator
                z = torch.randn(batch_size, Z_DIM).to(DEVICE)
                fake_imgs = generator(z)
                disc_fake = discriminator(fake_imgs.detach())
                disc_fake_loss = F.binary_cross_entropy_with_logits(
                    disc_fake, torch.zeros_like(disc_fake)
                )
                
                    # Fake images from NCA with breakthrough approach
                w = generator.mapping(z)
                    nca_seed = nca_model.get_seed(batch_size, IMG_SIZE, DEVICE, seed_type="distributed")
                    nca_output = nca_model(nca_seed, w, steps=NCA_STEPS_MIN, growth_boost=growth_boost)  # Apply growth boost
                nca_imgs = nca_model.to_rgb(nca_output)
                disc_nca = discriminator(nca_imgs.detach())
                disc_nca_loss = F.binary_cross_entropy_with_logits(
                    disc_nca, torch.zeros_like(disc_nca)
                )
                
                # Total discriminator loss
                disc_loss = (disc_real_loss + disc_fake_loss + disc_nca_loss) / 3
                disc_loss.backward()
                disc_optimizer.step()
                
                # Train Generator and NCA (only once per batch, after discriminator training)
                gen_optimizer.zero_grad()
                nca_optimizer.zero_grad()  # Zero NCA gradients
                
                # Generator adversarial loss
                gen_fake = discriminator(fake_imgs)
                gen_loss = F.binary_cross_entropy_with_logits(
                    gen_fake, torch.ones_like(gen_fake)
                )
                
                # NCA adversarial loss with growth monitoring
                nca_fake = discriminator(nca_imgs)
                nca_adversarial_loss = F.binary_cross_entropy_with_logits(
                    nca_fake, torch.ones_like(nca_fake)
                )
                
                # ADD RECONSTRUCTION LOSS FOR NCA (this was missing!)
                # NCA should learn to reconstruct target images, not just fool discriminator
                nca_reconstruction_loss = F.mse_loss(nca_imgs, real_imgs)
                
                # Combine adversarial and reconstruction losses for NCA
                nca_loss = nca_adversarial_loss + 0.5 * nca_reconstruction_loss
                
                # Monitor alive ratio and compute quality scores
                with torch.no_grad():
                    alive_mask = (nca_output[:, 3:4] > 0.1).float()
                    alive_ratio = alive_mask.mean().item()
                    
                    # Compute quality scores using evaluators
                    gen_quality_score = gen_evaluator(fake_imgs).mean().item()
                    nca_quality_score = nca_evaluator(nca_imgs).mean().item()
                
                # Train Transformer Critic
                transformer_optimizer.zero_grad()
                
                if transformer_critic.mode == "critic":
                    # Get quality scores for all images
                    real_score = transformer_critic(real_imgs)
                    fake_score = transformer_critic(fake_imgs.detach())
                    nca_score = transformer_critic(nca_imgs.detach())
                    
                    # Use discriminator scores as targets
                    with torch.no_grad():
                        real_target = torch.ones_like(real_score) * 0.9
                        fake_target = torch.sigmoid(gen_fake.detach())
                        nca_target = torch.sigmoid(nca_fake.detach())
                    
                    # Compute critic losses
                    transformer_real_loss = F.mse_loss(real_score, real_target)
                    transformer_fake_loss = F.mse_loss(fake_score, fake_target)
                    transformer_nca_loss = F.mse_loss(nca_score, nca_target)
                    
                    transformer_loss = (transformer_real_loss + transformer_fake_loss + transformer_nca_loss) / 3
                    
                else:  # imitator mode - EXTENDED GENERATION EXPERIMENTATION
                    # Generate imitations - transformer tries to improve generator and NCA outputs
                    fake_imitation, fake_imitation_loss = transformer_critic(fake_imgs.detach(), real_imgs)
                    nca_imitation, nca_imitation_loss = transformer_critic(nca_imgs.detach(), real_imgs)
                    
                    # ADAPTIVE GENERATION STRATEGY based on phase progress
                    exploration_factor = 1.0 - phase_progress  # More exploration early in generation phase
                    refinement_factor = phase_progress  # More refinement later in generation phase
                    
                    # Enhanced generation losses with adaptive weighting
                    # Early phase: Focus on exploration and diversity
                    # Later phase: Focus on refinement and quality
                    
                    # Direct reconstruction loss (weighted by refinement factor)
                    fake_reconstruction_loss = F.mse_loss(fake_imitation, real_imgs) * refinement_factor
                    nca_reconstruction_loss = F.mse_loss(nca_imitation, real_imgs) * refinement_factor
                    
                    # Perceptual loss using L1 distance (weighted by refinement factor)
                    fake_perceptual_loss = F.l1_loss(fake_imitation, real_imgs) * refinement_factor
                    nca_perceptual_loss = F.l1_loss(nca_imitation, real_imgs) * refinement_factor
                    
                    # Diversity/exploration loss (weighted by exploration factor)
                    # Encourage diverse outputs early in generation phase
                    if exploration_factor > 0.5:  # Early in generation phase
                        # Diversity loss: penalize outputs that are too similar to inputs
                        fake_diversity_loss = -F.l1_loss(fake_imitation, fake_imgs.detach()) * exploration_factor
                        nca_diversity_loss = -F.l1_loss(nca_imitation, nca_imgs.detach()) * exploration_factor
                    else:
                        fake_diversity_loss = torch.tensor(0.0, device=fake_imitation.device)
                        nca_diversity_loss = torch.tensor(0.0, device=nca_imitation.device)
                    
                    # Combined transformer loss with adaptive phase-based weighting
                    transformer_loss = (
                        0.3 * (fake_imitation_loss + nca_imitation_loss) / 2 +  # Base imitation loss
                        0.3 * (fake_reconstruction_loss + nca_reconstruction_loss) / 2 +  # Adaptive reconstruction
                        0.2 * (fake_perceptual_loss + nca_perceptual_loss) / 2 +  # Adaptive perceptual
                        0.2 * (fake_diversity_loss + nca_diversity_loss) / 2  # Adaptive diversity/exploration
                    )
                
                # DYNAMIC LEARNING RATE ADJUSTMENT during generation phase
                if phase_name == "GENERATION" and epochs_in_phase > 1:
                    # Create learning rate oscillation for experimentation
                    # Varies from 0.5x to 3x base rate during generation phase
                    lr_oscillation = 1.5 + 1.5 * torch.sin(torch.tensor(epochs_in_phase * 0.1)).item()
                    current_lr = transformer_optimizer.param_groups[0]['lr']
                    base_lr = LR * 0.5 * 2.0  # Base generation LR
                    target_lr = base_lr * lr_oscillation
                    
                    # Smooth LR transition (don't shock the optimizer)
                    for param_group in transformer_optimizer.param_groups:
                        param_group['lr'] = 0.9 * current_lr + 0.1 * target_lr
                
                transformer_loss.backward()
                transformer_optimizer.step()
                
                # Update generator and NCA with transformer feedback
                if transformer_critic.mode == "critic":
                    gen_transformer_score = transformer_critic(fake_imgs)
                    nca_transformer_score = transformer_critic(nca_imgs)
                    gen_loss += 0.1 * (1 - gen_transformer_score.mean())
                    nca_loss += 0.1 * (1 - nca_transformer_score.mean())
                
                gen_loss.backward()
                gen_optimizer.step()
                
                nca_loss.backward()
                nca_optimizer.step()  # Use NCA optimizer instead of optim_step
                
                # Update visuals every 100 batches
                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        # Get sample images
                        z = torch.randn(1, Z_DIM).to(DEVICE)
                        fake_sample = generator(z)
                        w = generator.mapping(z)
                        nca_seed = nca_model.get_seed(1, IMG_SIZE, DEVICE)
                        nca_output = nca_model(nca_seed, w, steps=NCA_STEPS_MIN)
                        nca_sample = nca_model.to_rgb(nca_output)
                        
                        # Get transformer output if in imitator mode
                        transformer_sample = None
                        # Enhanced transformer mode info with phase details
                        if is_isolated:
                            transformer_mode_info = f"{current_mode} (ISOLATED, epoch {epochs_in_phase}/30)"
                        else:
                            transformer_mode_info = f"{current_mode} ({phase_name}, epoch {epochs_in_phase})"
                        
                        if transformer_critic.mode == "imitator":
                            # Refine BOTH generator and NCA outputs
                            transformer_gen_sample, gen_loss = transformer_critic(fake_sample, real_imgs[0:1])
                            # Convert NCA output to RGB before passing to transformer
                            nca_rgb_sample = nca_model.to_rgb(nca_sample)  # Convert 8-channel NCA to 3-channel RGB
                            transformer_nca_sample, nca_loss = transformer_critic(nca_rgb_sample, real_imgs[0:1])
                            
                            # Alternate between showing gen and NCA refinements every few batches
                            show_nca_refinement = (batch_idx // 100) % 2 == 1
                            if show_nca_refinement:
                                transformer_sample = transformer_nca_sample
                                transformer_mode_info = f"imitator (refining NCA, loss: {nca_loss.item():.3f})"
                            else:
                                transformer_sample = transformer_gen_sample  
                                transformer_mode_info = f"imitator (refining Gen, loss: {gen_loss.item():.3f})"
                        
                        # Update UI with samples and scores
                        images_list = [
                            real_imgs[0],
                            fake_sample[0],
                            nca_sample[0],
                            transformer_sample[0] if transformer_sample is not None else None
                        ]
                        
                        update_status(
                            f"Epoch {epoch}, Batch {batch_idx} | Transformer: {phase_name} (C{cycle_num}, {epochs_in_phase}) | Growth: {growth_boost:.1f}x | Alive: {alive_ratio:.3f} | Gen: {gen_quality_score:.3f} | NCA: {nca_quality_score:.3f}",
                            images=images_list,
                            scores={
                                # === DISCRIMINATOR BREAKDOWN (ESSENTIAL FOR GAN HEALTH) ===
                                'disc_real_loss': disc_real_loss.item(),      # How well it identifies REAL images
                                'disc_fake_gen_loss': disc_fake_loss.item(),  # How well it identifies FAKE Generator images  
                                'disc_fake_nca_loss': disc_nca_loss.item(),   # How well it identifies FAKE NCA images
                                'disc_total_loss': disc_loss.item(),          # Combined discriminator loss
                                
                                # === GENERATOR/NCA ADVERSARIAL LOSSES ===
                                'gen_adversarial_loss': gen_loss.item(),      # Generator trying to fool discriminator
                                'nca_adversarial_loss': nca_adversarial_loss.item(),  # NCA adversarial component
                                'nca_reconstruction_loss': nca_reconstruction_loss.item(),  # NCA reconstruction component
                                'nca_total_loss': nca_loss.item(),            # Combined NCA loss
                                
                                # === LEGACY FIELDS (for compatibility) ===
                                'disc_loss': disc_loss.item(),
                                'gen_loss': gen_loss.item(),
                                'nca_loss': nca_loss.item(),
                                'transformer_loss': transformer_loss.item(),
                                
                                # === QUALITY EVALUATORS (INDEPENDENT) ===
                                'gen_quality': gen_quality_score,             # Independent quality assessment
                                'nca_quality': nca_quality_score,             # Independent quality assessment
                                
                                # === TRAINING HEALTH ===
                                'transformer_mode': transformer_mode_info,
                                'transformer_phase': phase_name,
                                'transformer_cycle': cycle_num,
                                'transformer_phase_progress': phase_progress,
                                'transformer_exploration_factor': exploration_factor if phase_name == "GENERATION" else 0.0,
                                'growth_boost': growth_boost,
                                'alive_ratio': alive_ratio,
                            }
                        )
            
            # Save checkpoint with metrics history
            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'gen_evaluator_state_dict': gen_evaluator.state_dict(),
                    'nca_evaluator_state_dict': nca_evaluator.state_dict(),
                    'transformer_critic_state_dict': transformer_critic.state_dict(),
                    'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                    'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                    'transformer_optimizer_state_dict': transformer_optimizer.state_dict(),
                    'nca_optimizer_state_dict': nca_optimizer.state_dict(),
                    'scores': getattr(update_status, 'metrics_history', [])  # Include metrics for charts
                }
                
                # Save both latest and epoch-specific checkpoint
                torch.save(checkpoint, latest_checkpoint)
                epoch_checkpoint = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
                torch.save(checkpoint, epoch_checkpoint)
                print(f"\n💾 Checkpoint saved at epoch {epoch} (with {len(getattr(update_status, 'metrics_history', []))} metrics)")
        
        print("Training completed!")
        return True
        
    except Exception as e:
        print(f"Error in training loop: {str(e)}")
        traceback.print_exc()
        return False

# --- Transformer Critic/Imitator ---
class TransformerCritic(nn.Module):
    def __init__(self, img_size=64, patch_size=8, dim=128, depth=4, heads=4, mlp_dim=256):
        super().__init__()
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.mode = "critic"  # Default mode
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))
        
        # Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim)
            for _ in range(depth)
        ])
        
        # Final layers for critic mode
        self.layer_norm = nn.LayerNorm(dim)
        self.critic_head = nn.Sequential(
            nn.Linear(dim * self.num_patches, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid()
        )
        
        # Imitator head for generation mode
        self.imitator_head = nn.Sequential(
            nn.Linear(dim * self.num_patches, mlp_dim * 2),
            nn.GELU(),
            nn.Linear(mlp_dim * 2, img_size * img_size * 3),
            nn.Tanh()
        )
    
    def set_mode(self, mode):
        """Switch between critic and imitator modes"""
        assert mode in ["critic", "imitator"], "Mode must be 'critic' or 'imitator'"
        self.mode = mode
        
    def forward(self, img, target=None):
        # Convert image to patches
        x = self.to_patch_embedding(img)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        x = self.transformer(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Flatten
        x = x.view(x.shape[0], -1)
        
        if self.mode == "critic":
            return self.critic_head(x)
        else:  # imitator mode
            if target is None:
                raise ValueError("Target image required for imitator mode")
            
            # Generate image
            generated = self.imitator_head(x)
            generated = generated.view(img.shape)  # Reshape to image format
            
            # Compute imitation loss
            imitation_loss = F.mse_loss(generated, target)
            
            return generated, imitation_loss

class EfficientTransformerBlock(nn.Module):
    """
    Efficient transformer block with optimizations from recent research.
    Reduces computational complexity while maintaining effectiveness.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm architecture for better gradient flow
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientAttention(nn.Module):
    """
    Efficient attention mechanism with linear complexity optimizations.
    Based on techniques from Performer, Linformer, and similar efficient transformers.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, attention_type="standard"):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_type = attention_type
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For efficient attention variants
        if attention_type == "linear":
            # Linear attention approximation
            self.feature_map = nn.ReLU()
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.attention_type == "linear":
            # Linear attention for O(N) complexity
            q = self.feature_map(q) + 1e-6
            k = self.feature_map(k) + 1e-6
            
            # Compute attention efficiently
            kv = torch.einsum('bhnd,bhne->bhde', k, v)
            qkv = torch.einsum('bhnd,bhde->bhne', q, kv)
            
            # Normalize
            k_sum = k.sum(dim=-2, keepdim=True)
            q_k_sum = torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(-2)).unsqueeze(-1)
            attn_output = qkv / (q_k_sum + 1e-6)
            
        else:
            # Standard attention with optimizations
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # Apply attention dropout only during training
            if self.training:
                attn = F.softmax(attn, dim=-1)
                attn = self.dropout(attn)
            else:
                attn = F.softmax(attn, dim=-1)
            
            attn_output = attn @ v
        
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        
        return attn_output

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head self attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class Rearrange(nn.Module):
    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        
    def forward(self, x):
        shape = x.shape
        if self.pattern == 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)':
            b, c, height, width = shape
            p1, p2 = self.axes_lengths['p1'], self.axes_lengths['p2']
            h, w = height // p1, width // p2
            
            # Reshape to extract patches
            x = x.view(b, c, h, p1, w, p2)
            # Permute dimensions to get the desired output shape
            x = x.permute(0, 2, 4, 3, 5, 1)
            # Combine patch dimensions and combine spatial dimensions
            x = x.reshape(b, h * w, p1 * p2 * c)
            
            return x
        else:
            raise NotImplementedError(f"Pattern {self.pattern} not implemented")

# --- Main Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the NCA-StyleGAN application.")
    parser.add_argument('--run-training', action='store_true', help='Run the training loop.')
    parser.add_argument('--test-checkpoint', action='store_true', help='Test checkpoint functionality.')
    parser.add_argument('--cleanup-checkpoints', action='store_true', help='Clean up old checkpoints.')
    parser.add_argument('--keep-checkpoints', type=int, default=KEEP_CHECKPOINTS, help=f'Number of recent checkpoints to keep (default: {KEEP_CHECKPOINTS})')
    args = parser.parse_args()

    if args.test_checkpoint:
        print("Testing checkpoint functionality...")
        # Create a simple model to test checkpoint saving
        test_model = nn.Linear(10, 5)
        test_optimizer = optim.Adam(test_model.parameters(), lr=0.001)
        
        # Create a test checkpoint
        test_checkpoint = {
            'epoch': 0,
            'model_state_dict': test_model.state_dict(),
            'optimizer_state_dict': test_optimizer.state_dict(),
            'test_data': [1, 2, 3, 4, 5]
        }
        
        # Print checkpoint directory information
        print(f"Checkpoint directory: {CHECKPOINT_DIR}")
        if os.path.exists(CHECKPOINT_DIR):
            print(f"Checkpoint directory exists. Contents: {os.listdir(CHECKPOINT_DIR)}")
            print(f"Permissions: {oct(os.stat(CHECKPOINT_DIR).st_mode & 0o777)}")
        else:
            print(f"Checkpoint directory does not exist. Creating it...")
            try:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True, mode=0o777)
                print(f"Created checkpoint directory. Permissions: {oct(os.stat(CHECKPOINT_DIR).st_mode & 0o777)}")
            except Exception as e:
                print(f"Error creating checkpoint directory: {str(e)}")
        
        # Save test checkpoint
        test_path = os.path.join(CHECKPOINT_DIR, 'test_checkpoint.pt')
        try:
            torch.save(test_checkpoint, test_path)
            print(f"Successfully saved test checkpoint to {test_path}")
            
            # Verify file exists and load it
            if os.path.exists(test_path):
                print(f"Test checkpoint exists with size: {os.path.getsize(test_path)} bytes")
                loaded = torch.load(test_path)
                print(f"Successfully loaded test checkpoint with keys: {list(loaded.keys())}")
            else:
                print("Error: Test checkpoint file not found after saving")
        except Exception as e:
            print(f"Error during checkpoint test: {str(e)}")
            # Try to create a simple test file
            try:
                test_file = os.path.join(CHECKPOINT_DIR, "test_write.txt")
                with open(test_file, "w") as f:
                    f.write("Test write access")
                print(f"Successfully created test file at {test_file}")
            except Exception as test_e:
                print(f"Error creating test file: {str(test_e)}")
    elif args.cleanup_checkpoints:
        print("Cleaning up old checkpoints...")
        print(f"Checkpoint directory: {CHECKPOINT_DIR}")
        if os.path.exists(CHECKPOINT_DIR):
            print(f"Contents before cleanup: {os.listdir(CHECKPOINT_DIR)}")
            cleanup_old_checkpoints(CHECKPOINT_DIR, args.keep_checkpoints)
            print(f"Contents after cleanup: {os.listdir(CHECKPOINT_DIR)}")
        else:
            print("Checkpoint directory does not exist.")
    elif args.run_training:
        print("Starting training worker...")
        training_loop()
    else:
        print("This script is now only for training. Use --run-training to start, --test-checkpoint to test, or --cleanup-checkpoints to clean up old checkpoints.")

# Advanced Homeostatic NCA with Memory and Attention
class HomeostaticNCA(nn.Module):
    def __init__(self, n_channels=16, n_memory_channels=8, n_hardware_channels=4, n_attention_heads=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_memory_channels = n_memory_channels  # Private memory not visible to neighbors
        self.n_hardware_channels = n_hardware_channels  # Immutable regulatory parameters
        self.n_attention_heads = n_attention_heads
        
        # Total channels: visible + memory + hardware
        total_channels = n_channels + n_memory_channels + n_hardware_channels
        
        # Perception kernels (Sobel filters for gradients)
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('identity', torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('laplacian', torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Perception network - processes local neighborhood
        perception_size = total_channels * 4  # 4 kernels per channel
        self.perception_net = nn.Sequential(
            nn.Linear(perception_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Attention mechanism for mode selection
        self.attention_embed = nn.Linear(n_hardware_channels, n_attention_heads)
        self.attention_temp = nn.Parameter(torch.tensor(1.0))
        
        # Multiple update pathways (attention heads)
        self.update_pathways = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_channels + n_memory_channels)  # Only update visible + memory, not hardware
            ) for _ in range(n_attention_heads)
        ])
        
        # Homeostatic controller
        self.homeostatic_controller = nn.Sequential(
            nn.Linear(n_memory_channels + 1, 16),  # +1 for alive_ratio
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),  # Output scaling factor
            nn.Sigmoid()
        )
        
        # Initialize with "do nothing" behavior
        for pathway in self.update_pathways:
            nn.init.zeros_(pathway[-1].weight)
            nn.init.zeros_(pathway[-1].bias)

    def perceive(self, x):
        """Apply perception kernels to all channels"""
        batch_size, channels, height, width = x.shape
        
        # Apply kernels to each channel
        perceived = []
        for kernel in [self.identity, self.sobel_x, self.sobel_y, self.laplacian]:
            kernel_expanded = kernel.expand(channels, -1, -1, -1)
            conv_result = F.conv2d(x, kernel_expanded, groups=channels, padding=1)
            perceived.append(conv_result)
        
        # Concatenate all perceptions
        perception = torch.cat(perceived, dim=1)  # [batch, channels*4, height, width]
        return perception

    def forward(self, x, step_prob=0.5):
        batch_size, total_channels, height, width = x.shape
        
        # Split channels
        visible = x[:, :self.n_channels]
        memory = x[:, self.n_channels:self.n_channels + self.n_memory_channels]
        hardware = x[:, -self.n_hardware_channels:]
        
        # Calculate alive cells (alpha > 0.1)
        alive_mask = (visible[:, 3:4] > 0.1).float()
        alive_ratio = alive_mask.mean(dim=[2, 3], keepdim=True)
        
        # Perception
        perception = self.perceive(x)  # [batch, total_channels*4, height, width]
        
        # Reshape for processing
        perception_flat = perception.view(batch_size, -1, height * width).permute(0, 2, 1)  # [batch, hw, features]
        memory_flat = memory.view(batch_size, -1, height * width).permute(0, 2, 1)  # [batch, hw, memory]
        hardware_flat = hardware.view(batch_size, -1, height * width).permute(0, 2, 1)  # [batch, hw, hardware]
        
        # Process perception
        processed_perception = self.perception_net(perception_flat)  # [batch, hw, 64]
        
        # Attention mechanism - hardware determines which pathways to activate
        attention_logits = self.attention_embed(hardware_flat) / self.attention_temp  # [batch, hw, n_heads]
        attention_weights = F.softmax(attention_logits, dim=-1)  # [batch, hw, n_heads]
        
        # Apply pathways with attention weighting
        pathway_outputs = []
        for i, pathway in enumerate(self.update_pathways):
            pathway_out = pathway(processed_perception)  # [batch, hw, channels+memory]
            pathway_outputs.append(pathway_out)
        
        pathway_stack = torch.stack(pathway_outputs, dim=-1)  # [batch, hw, channels+memory, n_heads]
        
        # Weighted combination of pathways
        attention_expanded = attention_weights.unsqueeze(-2)  # [batch, hw, 1, n_heads]
        weighted_update = (pathway_stack * attention_expanded).sum(dim=-1)  # [batch, hw, channels+memory]
        
        # Homeostatic regulation
        alive_ratio_expanded = alive_ratio.expand(-1, -1, height, width).view(batch_size, height * width, 1)
        memory_mean = memory_flat.mean(dim=-1, keepdim=True)  # Average memory state
        homeostatic_input = torch.cat([memory_mean, alive_ratio_expanded], dim=-1)
        homeostatic_scale = self.homeostatic_controller(homeostatic_input)  # [batch, hw, 1]
        
        # Apply homeostatic scaling
        weighted_update = weighted_update * homeostatic_scale
        
        # Reshape back to spatial dimensions
        update_spatial = weighted_update.permute(0, 2, 1).view(batch_size, -1, height, width)
        visible_update = update_spatial[:, :self.n_channels]
        memory_update = update_spatial[:, self.n_channels:]
        
        # Stochastic update mask (asynchronous updates)
        if self.training:
            update_mask = (torch.rand_like(visible_update[:, :1]) < step_prob).float()
        else:
            update_mask = torch.ones_like(visible_update[:, :1])
        
        # Apply updates only to visible and memory channels (hardware is immutable)
        new_visible = visible + visible_update * update_mask
        new_memory = memory + memory_update * update_mask
        
        # Enforce alive mask - dead cells have zero state
        alive_extended = F.max_pool2d(alive_mask, 3, stride=1, padding=1)
        new_visible = new_visible * alive_extended
        new_memory = new_memory * alive_extended
        
        # Recombine all channels
        new_x = torch.cat([new_visible, new_memory, hardware], dim=1)
        
        return new_x

def create_homeostatic_hardware(size, device):
    """Create specialized hardware configurations for different regions"""
    hardware = torch.zeros(1, 4, size, size, device=device)
    
    # Input region hardware (bottom-left)
    input_region = torch.zeros(4)
    input_region[0] = 1.0  # Input marker
    input_region[1] = 0.8  # High growth promotion
    input_region[2] = 0.2  # Low death promotion
    input_region[3] = 0.5  # Moderate stability
    
    # Growth region hardware (center)
    growth_region = torch.zeros(4)
    growth_region[0] = 0.0  # Not input
    growth_region[1] = 0.6  # Moderate growth
    growth_region[2] = 0.4  # Moderate death
    growth_region[3] = 0.8  # High stability preference
    
    # Boundary region hardware (edges)
    boundary_region = torch.zeros(4)
    boundary_region[0] = 0.0  # Not input
    boundary_region[1] = 0.2  # Low growth
    boundary_region[2] = 0.8  # High death (boundary cleanup)
    boundary_region[3] = 0.3  # Low stability
    
    # Apply hardware configurations
    # Input region (bottom-left quadrant)
    hardware[0, :, size//2:, :size//2] = input_region.view(-1, 1, 1)
    
    # Boundary regions (edges)
    hardware[0, :, :5, :] = boundary_region.view(-1, 1, 1)  # Top
    hardware[0, :, -5:, :] = boundary_region.view(-1, 1, 1)  # Bottom
    hardware[0, :, :, :5] = boundary_region.view(-1, 1, 1)  # Left
    hardware[0, :, :, -5:] = boundary_region.view(-1, 1, 1)  # Right
    
    # Growth region (center, overriding some boundary)
    center_start = size // 4
    center_end = 3 * size // 4
    hardware[0, :, center_start:center_end, center_start:center_end] = growth_region.view(-1, 1, 1)
    
    return hardware

def initialize_homeostatic_state(size, device):
    """Initialize state with visible channels, memory channels, and hardware"""
    n_channels = 16
    n_memory = 8
    n_hardware = 4
    
    # Initialize visible channels (RGBA + hidden)
    visible = torch.zeros(1, n_channels, size, size, device=device)
    
    # Add seed in center
    center = size // 2
    visible[0, 3, center-1:center+2, center-1:center+2] = 1.0  # Alpha channel
    visible[0, 4:, center-1:center+2, center-1:center+2] = torch.randn(n_channels-4, 3, 3, device=device) * 0.1
    
    # Initialize memory channels (internal cell memory)
    memory = torch.zeros(1, n_memory, size, size, device=device)
    
    # Create hardware configuration
    hardware = create_homeostatic_hardware(size, device)
    
    # Combine all channels
    state = torch.cat([visible, memory, hardware], dim=1)
    
    return state

def train_homeostatic_nca():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    size = 64
    n_channels = 16
    n_memory = 8
    n_hardware = 4
    
    # Initialize model
    model = HomeostaticNCA(n_channels=n_channels, n_memory_channels=n_memory, 
                          n_hardware_channels=n_hardware, n_attention_heads=4).to(device)
    
    # Load StyleGAN target if available
    target_path = "stylegan_target.pt"
    if os.path.exists(target_path):
        target = torch.load(target_path, map_location=device)
        print("Loaded StyleGAN target")
    else:
        # Create a simple target pattern for testing
        target = torch.zeros(1, n_channels, size, size, device=device)
        # Create a simple pattern - circle
        y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
        center = size // 2
        radius = size // 4
        circle_mask = ((x - center) ** 2 + (y - center) ** 2) < radius ** 2
        target[0, 0, circle_mask] = 1.0  # Red
        target[0, 3, circle_mask] = 1.0  # Alpha
        print("Created simple circle target")
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    # Training parameters
    n_steps = 64
    n_epochs = 1000
    
    losses = []
    alive_ratios = []
    
    print("Starting homeostatic NCA training...")
    
    for epoch in range(n_epochs):
        model.train()
        
        # Initialize state
        state = initialize_homeostatic_state(size, device)
        
        # Run CA for n_steps
        for step in range(n_steps):
            state = model(state, step_prob=0.5)
        
        # Calculate loss (only on visible channels)
        visible_state = state[:, :n_channels]
        loss = F.mse_loss(visible_state, target)
        
        # Calculate alive ratio for monitoring
        alive_mask = (visible_state[:, 3:4] > 0.1).float()
        alive_ratio = alive_mask.mean().item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        alive_ratios.append(alive_ratio)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Alive ratio = {alive_ratio:.4f}")
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss.item(),
                'alive_ratio': alive_ratio
            }, f'homeostatic_nca_checkpoint_epoch_{epoch}.pt')
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_state = initialize_homeostatic_state(size, device)
        states = [final_state[:, :n_channels].cpu()]
        
        for step in range(n_steps * 2):  # Run longer for evaluation
            final_state = model(final_state, step_prob=1.0)  # Deterministic for eval
            if step % 10 == 0:
                states.append(final_state[:, :n_channels].cpu())
    
    # Save results
    results = {
        'losses': losses,
        'alive_ratios': alive_ratios,
        'final_states': [s.numpy() for s in states[-5:]],  # Last 5 states
        'parameters': {
            'n_channels': n_channels,
            'n_memory': n_memory,
            'n_hardware': n_hardware,
            'size': size,
            'n_steps': n_steps,
            'n_epochs': n_epochs
        }
    }
    
    with open('homeostatic_nca_results.json', 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'final_states'}, f, indent=2)
    
    np.save('homeostatic_nca_states.npy', np.array([s.numpy() for s in states]))
    
    print(f"Training completed!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Final alive ratio: {alive_ratios[-1]:.4f}")
    print(f"Alive ratio range: {min(alive_ratios):.4f} - {max(alive_ratios):.4f}")
    
    return model, results

# Homeostatic NCA training function available for experimentation
# Uncomment the line below to run homeostatic NCA training instead of main training
# if __name__ == "__main__":
#     model, results = train_homeostatic_nca() 

# Add after the imports and before the training loop

def get_cyclical_transformer_mode(epoch):
    """
    Cyclical transformer training with extended generation phase for experimentation.
    
    🔄 CYCLE STRUCTURE (200 epochs per cycle):
    
    📊 CRITIC phase (Epochs 1-50, 25%):
       - Learn to evaluate image quality
       - Base learning rate
       - Focus: Quality assessment and discrimination
    
    🧠 ISOLATION phase (Epochs 51-80, 15%):
       - Consolidate learned patterns
       - Reduced learning rate (0.1x)
       - Focus: Knowledge consolidation without interference
    
    🎨 GENERATION phase (Epochs 81-200, 60% - EXTENDED!):
       - Generate and improve images
       - Dynamic learning rate (oscillating 0.5x to 3x)
       - Adaptive loss weighting (exploration → refinement)
       - Focus: Creative experimentation and generation
    
    ✨ Key Innovation: 120 epochs of generation experimentation per cycle!
    This gives the transformer extensive time to develop generation skills
    while cycling back to criticism and consolidation regularly.
    """
    cycle_length = 200  # 200-epoch cycles
    position_in_cycle = (epoch - 1) % cycle_length + 1  # 1-indexed position in cycle
    cycle_number = (epoch - 1) // cycle_length + 1
    
    if position_in_cycle <= 50:      # First 25% - CRITIC phase
        mode = "critic"
        isolated = False
        phase_name = "CRITIC"
        phase_progress = position_in_cycle / 50.0
        
    elif position_in_cycle <= 80:    # Next 15% - ISOLATION phase  
        mode = "critic" 
        isolated = True
        phase_name = "ISOLATION"
        phase_progress = (position_in_cycle - 50) / 30.0
        
    else:                           # Final 60% - GENERATION phase (EXTENDED!)
        mode = "imitator"
        isolated = False
        phase_name = "GENERATION"
        phase_progress = (position_in_cycle - 80) / 120.0
    
    return {
        'mode': mode,
        'isolated': isolated,
        'phase_name': phase_name,
        'phase_progress': phase_progress,
        'cycle_number': cycle_number,
        'position_in_cycle': position_in_cycle,
        'epochs_in_phase': position_in_cycle if position_in_cycle <= 50 else (position_in_cycle - 50 if position_in_cycle <= 80 else position_in_cycle - 80)
    }