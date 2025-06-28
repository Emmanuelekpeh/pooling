#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import signal
import sys
import argparse
import time
import base64
import io
import numpy as np
from datetime import datetime
import random
import math
import torchvision
from tqdm import tqdm

# === FIXED TRAINING CONFIGURATION ===
# Based on mode collapse research and diagnostic results
# KEY FIX: Train GENERATOR/NCA more frequently, DISCRIMINATOR less frequently
# This addresses mode collapse by preventing discriminator from overpowering generators

# Core settings
IMG_SIZE = 64
BATCH_SIZE = 4
Z_DIM = 64
W_DIM = 128
NCA_STEPS_MIN = 8
NCA_STEPS_MAX = 16
EPOCHS = 6500

# CRITICAL LEARNING RATE FIXES
BASE_LR = 0.0002
GEN_LR = BASE_LR * 0.25    # Much lower to prevent collapse
DISC_LR = BASE_LR * 1.0    # Standard discriminator LR
NCA_LR = BASE_LR * 0.1     # Even lower for stable NCA growth
TRANSFORMER_LR = BASE_LR * 0.5  # Moderate for transformer

# TRAINING BALANCE - CORRECTED FOR MODE COLLAPSE
GEN_STEPS = 2              # Train generator MORE frequently  
NCA_STEPS_TRAIN = 2        # Train NCA MORE frequently
DISC_STEPS_EARLY = 1       # Train discriminator LESS in early epochs (was 3 - wrong!)
DISC_STEPS_LATE = 2        # Keep discriminator training minimal (was 2 - wrong!)
EARLY_EPOCH_THRESHOLD = 50

# REGULARIZATION SETTINGS
NOISE_INJECTION = 0.05     # Noise to real images
LABEL_SMOOTHING = 0.9      # Use 0.9 instead of 1.0 for real labels
GRADIENT_PENALTY_WEIGHT = 10

# TRANSFORMER ISOLATION
TRANSFORMER_START_EPOCH = 25
TRANSFORMER_ISOLATION_EPOCHS = 50

# Paths
DATA_DIR = './data/ukiyo-e-small'
CHECKPOINT_DIR = './checkpoints'
SAMPLES_DIR = './samples'
STATUS_FILE = os.path.join(SAMPLES_DIR, 'status.json')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")

# === IMPROVED GENERATOR ===

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, w_dim)
        )

    def forward(self, z, truncation=True):
        w = self.mapping(z)
        if truncation and self.training:
            w = torch.clamp(w, -2.0, 2.0)
        return w

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (2.0 / in_dim) ** 0.5

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style = EqualizedLinear(w_dim, channels * 2)

    def forward(self, x, w):
        style = self.style(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        return gamma * self.norm(x) + beta

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = (2.0 / (in_channels * kernel_size * kernel_size)) ** 0.5
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.noise1 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2)
        
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        self.noise2 = NoiseInjection(out_channels)
        self.adain2 = AdaIN(out_channels, w_dim)

    def forward(self, x, w):
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.adain1(x, w)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.adain2(x, w)
        x = self.activation(x)
        return x

class IntegratedGenerator(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        self.mapping = MappingNetwork(z_dim, w_dim)
        
        # Start with 4x4 feature map
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.initial_block = GeneratorBlock(512, 512, w_dim)
        
        # Upsampling blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block_8x8 = GeneratorBlock(512, 256, w_dim)
        self.block_16x16 = GeneratorBlock(256, 128, w_dim)
        self.block_32x32 = GeneratorBlock(128, 64, w_dim)
        self.block_64x64 = GeneratorBlock(64, 32, w_dim)
        
        self.to_rgb = EqualizedConv2d(32, 3, 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, EqualizedConv2d, nn.Linear, EqualizedLinear)):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, noise, return_w=False):
        w = self.mapping(noise)
        
        x = self.const.repeat(noise.size(0), 1, 1, 1)
        x = self.initial_block(x, w)
        
        x = self.upsample(x) # 4x4 -> 8x8
        x = self.block_8x8(x, w)
        
        x = self.upsample(x) # 8x8 -> 16x16
        x = self.block_16x16(x, w)
        
        x = self.upsample(x) # 16x16 -> 32x32
        x = self.block_32x32(x, w)
        
        x = self.upsample(x) # 32x32 -> 64x64
        x = self.block_64x64(x, w)
        
        rgb = self.to_rgb(x)
        rgb = torch.tanh(rgb)
        
        if return_w:
            return rgb, w
        return rgb

# === IMPROVED NCA ===

class SimpleGrowthNCA(nn.Module):
    def __init__(self, n_channels=16, w_dim=128):
        super().__init__()
        self.n_channels = n_channels
        self.w_dim = w_dim
        
        # The update network should be composed of Linear layers, not Conv2d,
        # because it processes a flattened list of cell states.
        self.update_net = nn.Sequential(
            nn.Linear(n_channels * 3 + w_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_channels)
        )

        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # W-space conditioning with stability focus
        self.w_processor = nn.Sequential(
            nn.Linear(w_dim, w_dim),
            nn.LayerNorm(w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim)
        )
        
        # Stability predictor (like StableNCA)
        self.stability_predictor = nn.Sequential(
            nn.Linear(w_dim + 3, 64),  # w + [alive_ratio, growth_rate, variance]
            nn.ReLU(),
            nn.Linear(64, 3),  # [growth_scale, death_scale, stability_scale]
            nn.Sigmoid()
        )
        
        # Style modulation (single layer, reduced complexity)
        self.style_mod = nn.Linear(w_dim, n_channels * 2)
        
        # Growth parameters
        self.base_growth_rate = nn.Parameter(torch.tensor(0.2))
        self.base_threshold = nn.Parameter(torch.tensor(0.1))
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.1)  # Smaller initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def perceive(self, x):
        """Simple perception with gradient control"""
        sobel_x_expanded = self.sobel_x.repeat(self.n_channels, 1, 1, 1)
        sobel_y_expanded = self.sobel_y.repeat(self.n_channels, 1, 1, 1)
        
        dx = F.conv2d(x, sobel_x_expanded, groups=self.n_channels, padding=1)
        dy = F.conv2d(x, sobel_y_expanded, groups=self.n_channels, padding=1)
        
        return torch.cat([x, dx, dy], dim=1)

    def get_seed(self, batch_size, size, device, seed_type="distributed"):
        """Create initial seed with controlled alpha values to prevent immortality"""
        seed = torch.zeros(batch_size, self.n_channels, size, size, device=device)
        
        if seed_type == "center":
            center = size // 2
            radius = 3
            # Create circular seed
            y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
            dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
            mask = dist <= radius
            
            for b in range(batch_size):
                # Set RGB channels with some color - moderate initial values
                seed[b, 0, mask] = 0.3  # Red
                seed[b, 1, mask] = 0.2  # Green
                seed[b, 2, mask] = 0.4  # Blue
                seed[b, 3, mask] = 1.0  # Alpha - fully opaque
                
                # Set other channels to small random values in the living region
                seed[b, 4:, mask] = torch.randn(self.n_channels - 4, mask.sum(), device=device) * 0.1
        
        elif seed_type == "distributed":
            num_seeds = 3
            radius = 2
            margin = size // 6  # Keep seeds away from edges
            
            # Get distributed positions for seeds
            positions = []
            for i in range(num_seeds):
                x = margin + torch.randint(size - 2*margin, (1,), device=device)
                y = margin + torch.randint(size - 2*margin, (1,), device=device)
                positions.append((x.item(), y.item()))
            
            # Place seeds at distributed positions
            for b in range(batch_size):
                for pos_x, pos_y in positions:
                    # Create circular mask for this seed
                    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
                    dist = ((x - pos_x) ** 2 + (y - pos_y) ** 2) ** 0.5
                    mask = dist <= radius
                    
                    # Set RGB channels with some color variation
                    seed[b, 0, mask] = 0.3 + torch.rand(1, device=device) * 0.2  # Red
                    seed[b, 1, mask] = 0.2 + torch.rand(1, device=device) * 0.2  # Green
                    seed[b, 2, mask] = 0.4 + torch.rand(1, device=device) * 0.2  # Blue
                    seed[b, 3, mask] = 1.0  # Alpha - fully opaque
                    
                    # Set other channels to small random values in the living region
                    seed[b, 4:, mask] = torch.randn(self.n_channels - 4, mask.sum(), device=device) * 0.1
        
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
        alpha = torch.sigmoid(alpha)  # Ensure alpha is in [0, 1]
        
        return torch.cat([rgb, alpha], dim=1)

    def get_stability_params(self, x, w, prev_alive_ratio=None):
        """Get stability parameters based on current state"""
        # Calculate population metrics
        alive_mask = (x[:, 3:4] > self.base_threshold).float()
        alive_ratio = alive_mask.mean(dim=[2, 3], keepdim=True)
        
        # Calculate growth rate and variance
        if prev_alive_ratio is not None:
            growth_rate = alive_ratio - prev_alive_ratio
            variance = alive_mask.var(dim=[2, 3], keepdim=True)
        else:
            growth_rate = torch.zeros_like(alive_ratio)
            variance = torch.zeros_like(alive_ratio)
        
        # Prepare stability input
        stability_input = torch.cat([
            w,
            alive_ratio.squeeze(-1).squeeze(-1),
            growth_rate.squeeze(-1).squeeze(-1),
            variance.squeeze(-1).squeeze(-1)
        ], dim=1)
        
        # Get stability parameters
        stability_params = self.stability_predictor(stability_input)
        return {
            'growth_scale': stability_params[:, 0:1],
            'death_scale': stability_params[:, 1:2],
            'stability_scale': stability_params[:, 2:3],
            'alive_ratio': alive_ratio,
            'growth_rate': growth_rate
        }

    def forward(self, x, w, steps, growth_boost=1.0):
        for _ in range(steps):
            # Perception: Get state of cell and its neighbors
            perceived_x = self.perceive(x)
            perceived_flat = perceived_x.permute(0, 2, 3, 1).reshape(-1, self.n_channels * 3)
            
            # Expand w to provide guidance to each cell
            w_expanded = w.repeat_interleave(perceived_flat.shape[0] // w.shape[0], dim=0)

            # Combine perception with W-vector guidance
            update_input = torch.cat([perceived_flat, w_expanded], dim=1)
            
            # Get update vector
            ds = self.update_net(update_input)
            ds = ds.reshape(x.shape[0], x.shape[2], x.shape[3], self.n_channels)
            ds = ds.permute(0, 3, 1, 2)
            
            # Get living mask
            alive_mask = (x[:, 3:4, :, :] > 0.1).float()
            
            # Create an update mask that includes living cells AND their immediate neighbors.
            update_mask = F.max_pool2d(alive_mask, kernel_size=3, stride=1, padding=1)

            # Apply update only to cells that can be affected
            x = x + ds * update_mask
        
        return x

# === IMPROVED DISCRIMINATOR ===

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.Conv2d(256, 512, 4, 2, 1),
        ])
        
        self.final = nn.Conv2d(512, 1, 2)
        self.dropout = nn.Dropout(0.3)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img, return_features=False):
        features = []
        x = img
        
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)
            if return_features:
                features.append(x)
        
        final_out = self.final(x)
        final_out_flat = final_out.view(x.size(0), -1)
            
        if return_features:
            return final_out_flat, features
        return final_out_flat

# === CROSS-EVALUATION NETWORKS ===

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

# === TRANSFORMER COMPONENTS ===

class EfficientTransformerBlock(nn.Module):
    """Efficient transformer block with pre-norm architecture"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm architecture for better gradient flow
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientAttention(nn.Module):
    """Efficient attention mechanism with linear complexity"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, attention_type="standard"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Efficient attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

# === TRANSFORMER CRITIC ===

class TransformerCritic(nn.Module):
    """
    Transformer-based critic that evolves into an imitator after epoch 350.
    Based on efficient transformer principles from recent research.
    """
    def __init__(self, img_size=64, patch_size=8, embed_dim=256, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.mode = "critic"  # "critic" or "imitator"
        
        # Patch embedding - convert image patches to tokens
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Efficient transformer blocks with reduced complexity
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Critic head (for evaluation mode)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Imitator head (for generation mode)
        self.imitator_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, self.patch_size * self.patch_size * 3),  # Correct output size
            nn.Tanh()
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights following transformer best practices"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def set_mode(self, mode):
        """Switch between critic, imitator, isolation, and dormant modes"""
        assert mode in ["critic", "imitator", "isolation", "dormant"], f"Mode must be one of 'critic', 'imitator', 'isolation', 'dormant', but got '{mode}'"
        self.mode = mode
        
    def forward(self, x, target=None):
        B, C, H, W = x.shape
        
        # Convert to patches and embed
        patches = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if self.mode == "critic":
            # Use CLS token for criticism
            cls_output = x[:, 0]  # [B, embed_dim]
            quality_score = self.critic_head(cls_output)
            return quality_score.squeeze(-1)
            
        elif self.mode == "imitator" or self.mode == "isolation":
            # Use all patch tokens for generation
            if target is None:
                raise ValueError("Target image required for imitator/isolation mode")
            
            # Extract patch features (excluding CLS token)
            patch_features = x[:, 1:]  # [B, num_patches, embed_dim]
            
            # Generate imitation
            generated_patches = self.imitator_head(patch_features)  # [B, num_patches, patch_size*patch_size*3]
            
            # Reshape to image format
            # Reshape to [B, num_patches, 3, patch_size, patch_size]
            generated_patches = generated_patches.view(B, self.num_patches, 3, self.patch_size, self.patch_size)
            
            # Rearrange from patches to image [B, C, H, W]
            patches_per_side = int(self.num_patches ** 0.5)
            generated_img = torch.zeros(B, 3, H, W, device=x.device)
            for i in range(B):
                for j in range(self.num_patches):
                    row = (j // patches_per_side) * self.patch_size
                    col = (j % patches_per_side) * self.patch_size
                    generated_img[i, :, row:row+self.patch_size, col:col+self.patch_size] = generated_patches[i, j]

            # Compute imitation loss
            imitation_loss = F.mse_loss(generated_img, target)
            
            return generated_img, imitation_loss

# === DATASET ===

class ImageDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.images = []
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

# === TRAINING UTILITIES ===

def get_progressive_growth_boost(epoch, total_epochs=600):
    if epoch < total_epochs * 0.2:
        return 2.5
    elif epoch < total_epochs * 0.4:
        return 2.0
    elif epoch < total_epochs * 0.6:
        return 1.5
    else:
        return 1.0

def get_cyclical_transformer_mode(epoch):
    """Get transformer mode based on a 3-phase cycle: critic, isolation, imitator."""
    if epoch < TRANSFORMER_START_EPOCH:
        return 'dormant'

    # The cycle starts after the dormant period
    effective_epoch = epoch - TRANSFORMER_START_EPOCH
    cycle_length = 90  # 30 epochs for critic, 30 for isolation, 30 for imitator
    phase = effective_epoch % cycle_length

    if phase < 30:
        return "critic"
    elif phase < 60:
        return "isolation"
    else:
        return "imitator"

def gradient_penalty(discriminator, real_imgs, fake_imgs, device):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolated.requires_grad_(True)
    
    disc_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty

def save_checkpoint(epoch, models, optimizers, metrics_history):
    """Save training checkpoint with proper error handling"""
    try:
        checkpoint = {
            'epoch': epoch,
            'models': {
                'generator': models['generator'].state_dict(),
                'discriminator': models['discriminator'].state_dict(),
                'nca': models['nca'].state_dict(),
                'transformer': models['transformer'].state_dict()
            },
            'optimizers': {
                'generator': optimizers['generator'].state_dict(),
                'discriminator': optimizers['discriminator'].state_dict(),
                'nca': optimizers['nca'].state_dict(),
                'transformer': optimizers['transformer'].state_dict()
            },
            'metrics_history': metrics_history
        }
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_checkpoint_path)
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(keep_last_n=5)
        
        print(f"💾 Latest checkpoint saved at epoch {epoch}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving checkpoint: {str(e)}")
        return False

def load_checkpoint(checkpoint_path, models, optimizers):
    """Load checkpoint, gracefully skipping mismatched keys."""
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found, starting from scratch.")
        return 0, []

    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        for name, model in models.items():
            if name in checkpoint.get('models', {}):
                try:
                    model.load_state_dict(checkpoint['models'][name], strict=False)
                    print(f"✅ Loaded {name} model state (non-strict)")
                except TypeError: # For older checkpoints
                     model.load_state_dict(checkpoint[f'{name}_state_dict'], strict=False)
                     print(f"✅ Loaded {name} model state from legacy key (non-strict)")
                except Exception as e:
                    print(f"⚠️ Could not load {name} model state: {e}")
            else:
                print(f"❌ No state found for {name} model in checkpoint.")

        for name, optimizer in optimizers.items():
            if name in checkpoint.get('optimizers', {}):
                try:
                    optimizer.load_state_dict(checkpoint['optimizers'][name])
                    print(f"✅ Loaded {name} optimizer state")
                except TypeError: # For older checkpoints
                    optimizer.load_state_dict(checkpoint[f'opt_{name}_state_dict'])
                    print(f"✅ Loaded {name} optimizer state from legacy key")
                except Exception as e:
                    print(f"⚠️ Could not load {name} optimizer state: {e}")
            else:
                print(f"❌ No state found for {name} optimizer in checkpoint.")

        start_epoch = checkpoint.get('epoch', 0)
        metrics_history = checkpoint.get('metrics_history', [])
        
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch, metrics_history

    except Exception as e:
        import traceback
        print(f"❌ Major error loading checkpoint: {e}")
        traceback.print_exc()
        return 0, []

def cleanup_old_checkpoints(keep_last_n=5):
    """Clean up old epoch-specific checkpoints, keeping only the last N"""
    try:
        if not os.path.exists(CHECKPOINT_DIR):
            return
            
        # Get all epoch checkpoint files
        checkpoint_files = []
        for filename in os.listdir(CHECKPOINT_DIR):
            if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                try:
                    epoch_num = int(filename.replace('checkpoint_epoch_', '').replace('.pt', ''))
                    filepath = os.path.join(CHECKPOINT_DIR, filename)
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
                    print(f"🗑️ Removed old checkpoint: checkpoint_epoch_{epoch_num}.pt")
                except Exception as e:
                    print(f"⚠️ Error removing {filepath}: {str(e)}")
            
            print(f"✅ Cleanup complete: Kept {min(len(checkpoint_files), keep_last_n)} most recent checkpoints")
        
    except Exception as e:
        print(f"❌ Error during checkpoint cleanup: {str(e)}")

def signal_handler(signum, frame):
    global interrupted, current_models, current_optimizers, current_epoch, metrics_history
    print("\n🛑 Graceful interruption requested... finishing current batch")
    interrupted = True
    
    # Save temporary checkpoint if we have models initialized
    if current_models is not None and current_optimizers is not None and current_epoch is not None:
        print("💾 Saving emergency checkpoint...")
        try:
            save_checkpoint(current_epoch, current_models, current_optimizers, metrics_history)
            print("✅ Emergency checkpoint saved successfully")
        except Exception as e:
            print(f"⚠️ Failed to save emergency checkpoint: {str(e)}")

def tensor_to_b64(tensor):
    """Converts the first tensor in a batch to a base64 encoded PNG image."""
    try:
        # Ensure tensor is on CPU and detached from the computation graph
        tensor = tensor.detach().cpu()
        
        # Check for invalid values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: tensor contains NaN or Inf values, creating black image")
            # Create a black image as fallback
            tensor = torch.zeros_like(tensor)
        
        # Take only the first image from the batch and normalize it
        single_img = tensor[0]  # Shape: [3, 64, 64]
        
        # Check tensor shape
        if single_img.dim() != 3 or single_img.shape[0] != 3:
            print(f"Warning: unexpected tensor shape {single_img.shape}, expected [3, H, W]")
            # Create a black image with correct shape
            single_img = torch.zeros(3, 64, 64)
        
        # Normalize from [-1, 1] to [0, 1]
        single_img = (single_img + 1.0) / 2.0
        # Clamp and convert to uint8
        ndarr = single_img.mul(255).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        # Create PIL Image and resize to make it larger
        im = Image.fromarray(ndarr)
        # Resize with nearest neighbor for crisp pixels
        im = im.resize((256, 256), Image.NEAREST)
        # Save to buffer
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error in tensor_to_b64: {str(e)}")
        # Create a solid black image as fallback
        black_img = Image.new('RGB', (256, 256), color='black')
        buffer = io.BytesIO()
        black_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def update_status(message, images=None, scores=None):
    """Update the status file with current training state"""
    status = {
        'status': message,
        'error': False
    }
    
    if images is not None:
        # Convert images to base64
        b64_images = []
        for img in images:
            if img is not None:
                img = img.detach().cpu()
                if len(img.shape) == 4:
                    img = img[0]  # Take first image from batch
                img = torch.clamp(img, -1, 1)
                img = ((img + 1) * 127.5).to(torch.uint8)
                buffer = io.BytesIO()
                torchvision.utils.save_image(img.float() / 255.0, buffer, format='PNG')
                b64_images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
            else:
                b64_images.append(None)
        status['images'] = b64_images
    
    if scores is not None:
        status['scores'] = scores
    
    # Save status
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f)

# === CROSS-LEARNING SYSTEM ===

class QualityAssessment(nn.Module):
    """Assess image quality using multiple perceptual metrics"""
    def __init__(self):
        super().__init__()
        # Simple quality metrics - can be expanded
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
    def edge_density(self, img):
        """Measure edge density as quality indicator"""
        gray = img.mean(dim=1, keepdim=True)  # Convert to grayscale
        
        # Apply Sobel filters
        edges_x = F.conv2d(gray, self.sobel_x, padding=1)
        edges_y = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        return edges.mean(dim=[1, 2, 3])  # Average edge strength per image
    
    def color_diversity(self, img):
        """Measure color diversity"""
        # Standard deviation across color channels
        color_std = img.std(dim=[2, 3])  # Std per channel per image
        return color_std.mean(dim=1)  # Average across channels
    
    def spatial_coherence(self, img):
        """Measure spatial smoothness/coherence"""
        # Gradient magnitude
        grad_x = torch.diff(img, dim=3)  # Horizontal gradients
        grad_y = torch.diff(img, dim=2)  # Vertical gradients
        
        # Pad to maintain size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        gradient_mag = torch.sqrt(grad_x**2 + grad_y**2)
        return 1.0 / (1.0 + gradient_mag.mean(dim=[1, 2, 3]))  # Inverse - higher is smoother
    
    def forward(self, img):
        """Compute comprehensive quality score"""
        edge_score = self.edge_density(img)
        color_score = self.color_diversity(img)
        coherence_score = self.spatial_coherence(img)
        
        # Weighted combination
        quality = 0.4 * edge_score + 0.3 * color_score + 0.3 * coherence_score
        return quality, {
            'edge_density': edge_score,
            'color_diversity': color_score,
            'spatial_coherence': coherence_score
        }

class CrossLearningSystem(nn.Module):
    """Smart cross-learning system using quality feedback"""
    def __init__(self):
        super().__init__()
        self.quality_assessor = QualityAssessment()
        
        # Learning weights - start equal, adapt based on performance
        self.register_buffer('gen_weight', torch.tensor(1.0))
        self.register_buffer('nca_weight', torch.tensor(1.0))
        
        # Performance history for adaptive weighting
        self.gen_quality_history = []
        self.nca_quality_history = []
        self.history_length = 20
        
    def assess_outputs(self, gen_imgs, nca_imgs, real_imgs):
        """Assess quality of generated images"""
        with torch.no_grad():
            gen_quality, gen_metrics = self.quality_assessor(gen_imgs)
            nca_quality, nca_metrics = self.quality_assessor(nca_imgs)
            real_quality, real_metrics = self.quality_assessor(real_imgs)
            
            # Compare to real images
            gen_similarity = 1.0 - torch.abs(gen_quality - real_quality)
            nca_similarity = 1.0 - torch.abs(nca_quality - real_quality)
            
            return {
                'gen_quality': gen_quality.mean().item(),
                'nca_quality': nca_quality.mean().item(),
                'real_quality': real_quality.mean().item(),
                'gen_similarity': gen_similarity.mean().item(),
                'nca_similarity': nca_similarity.mean().item(),
                'gen_metrics': {k: v.mean().item() for k, v in gen_metrics.items()},
                'nca_metrics': {k: v.mean().item() for k, v in nca_metrics.items()}
            }
    
    def update_learning_weights(self, assessment):
        """Adapt learning weights based on performance"""
        self.gen_quality_history.append(assessment['gen_similarity'])
        self.nca_quality_history.append(assessment['nca_similarity'])
        
        # Keep history bounded
        if len(self.gen_quality_history) > self.history_length:
            self.gen_quality_history.pop(0)
        if len(self.nca_quality_history) > self.history_length:
            self.nca_quality_history.pop(0)
        
        if len(self.gen_quality_history) >= 5:  # Need some history
            gen_trend = sum(self.gen_quality_history[-5:]) / 5
            nca_trend = sum(self.nca_quality_history[-5:]) / 5
            
            # Boost the better performer slightly
            total = gen_trend + nca_trend + 1e-8
            self.gen_weight = torch.tensor(0.5 + 0.3 * (gen_trend / total))
            self.nca_weight = torch.tensor(0.5 + 0.3 * (nca_trend / total))
    
    def compute_cross_learning_loss(self, gen_imgs, nca_imgs, real_imgs):
        """Compute cross-learning loss to improve both models"""
        # Quality-based reconstruction loss
        gen_quality, _ = self.quality_assessor(gen_imgs)
        nca_quality, _ = self.quality_assessor(nca_imgs)
        real_quality, _ = self.quality_assessor(real_imgs)
        
        # Both should match real quality
        gen_quality_loss = F.mse_loss(gen_quality, real_quality)
        nca_quality_loss = F.mse_loss(nca_quality, real_quality)
        
        # Cross-model consistency (when both are good)
        quality_threshold = 0.3  # Only enforce consistency for decent quality
        good_gen_mask = (gen_quality > quality_threshold).float()
        good_nca_mask = (nca_quality > quality_threshold).float()
        
        consistency_mask = good_gen_mask * good_nca_mask
        if consistency_mask.sum() > 0:
            # Encourage consistency between good outputs
            consistency_loss = F.mse_loss(
                gen_imgs * consistency_mask.view(-1, 1, 1, 1),
                nca_imgs * consistency_mask.view(-1, 1, 1, 1)
            )
        else:
            consistency_loss = torch.tensor(0.0, device=gen_imgs.device)
        
        return {
            'gen_quality_loss': gen_quality_loss,
            'nca_quality_loss': nca_quality_loss,
            'consistency_loss': consistency_loss,
            'total_cross_loss': gen_quality_loss + nca_quality_loss + 0.1 * consistency_loss
        }
    
    def forward(self, gen_imgs, nca_imgs, real_imgs):
        """Full cross-learning forward pass"""
        assessment = self.assess_outputs(gen_imgs, nca_imgs, real_imgs)
        cross_losses = self.compute_cross_learning_loss(gen_imgs, nca_imgs, real_imgs)
        self.update_learning_weights(assessment)
        
        return assessment, cross_losses

# === MAIN TRAINING FUNCTION ===

def training_loop():
    # Model and optimizer setup
    models = {
        'generator': IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE),
        'discriminator': Discriminator(IMG_SIZE).to(DEVICE),
        'nca': SimpleGrowthNCA(n_channels=16, w_dim=W_DIM).to(DEVICE),
        'transformer': TransformerCritic(img_size=IMG_SIZE).to(DEVICE)
    }

    optimizers = {
        'generator': optim.Adam(models['generator'].parameters(), lr=GEN_LR, betas=(0.5, 0.999)),
        'discriminator': optim.Adam(models['discriminator'].parameters(), lr=DISC_LR, betas=(0.5, 0.999)),
        'nca': optim.Adam(models['nca'].parameters(), lr=NCA_LR, betas=(0.5, 0.999)),
        'transformer': optim.Adam(models['transformer'].parameters(), lr=TRANSFORMER_LR, betas=(0.5, 0.999))
    }

    # Load checkpoint
    checkpoint_path = CHECKPOINT_PATH
    start_epoch, metrics_history = load_checkpoint(checkpoint_path, models, optimizers)
    if not isinstance(metrics_history, list):
        metrics_history = []

    # Data loading
    dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for stability
        pin_memory=False,  # Not needed without multiprocessing
        drop_last=True
    )
    
    # Print training configuration
    print("🔧 FIXED TRAINING SYSTEM")
    print("============================================================")
    print("🎯 Addressing mode collapse, transformer interference, and training stability")
    print("📊 Generator LR: 5e-05 (reduced for stability)")
    print("📊 Discriminator LR: 0.0002 (standard)")
    print("📊 NCA LR: 2e-05 (very low for growth)")
    print("📊 Transformer LR: 0.0001 (isolated)")
    print("============================================================")
    
    print(f"📊 Dataset size: {len(dataset)}")
    print("🚀 Training started - Press Ctrl+C to stop gracefully")
    
    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        metrics_history.append({})
        
        # Set transformer mode
        transformer_mode = get_cyclical_transformer_mode(epoch)
        models['transformer'].set_mode(transformer_mode)
        
        # Determine training intervals
        gen_train_interval = 1
        nca_train_interval = 1
        disc_train_interval = DISC_STEPS_EARLY if epoch < EARLY_EPOCH_THRESHOLD else DISC_STEPS_LATE

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, real_imgs in progress_bar:
            real_imgs = real_imgs.to(DEVICE)

            # ---------------------------------
            # Train Discriminator
            # ---------------------------------
            if batch_idx % disc_train_interval == 0:
                models['discriminator'].zero_grad()
                
                # On real images
                d_real_pred, _ = models['discriminator'](real_imgs, return_features=True)
                real_labels = torch.full_like(d_real_pred, LABEL_SMOOTHING)
                loss_d_real = F.binary_cross_entropy_with_logits(d_real_pred, real_labels)
                
                # On fake images (Generator)
                with torch.no_grad():
                    noise = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
                    fake_imgs_gen, w = models['generator'](noise, return_w=True)
                d_fake_gen_pred, _ = models['discriminator'](fake_imgs_gen.detach(), return_features=True)
                loss_d_fake_gen = F.binary_cross_entropy_with_logits(d_fake_gen_pred, torch.zeros_like(d_fake_gen_pred))

                # On fake images (NCA)
                with torch.no_grad():
                    seed = models['nca'].get_seed(BATCH_SIZE, IMG_SIZE, DEVICE)
                    nca_steps = random.randint(NCA_STEPS_MIN, NCA_STEPS_MAX)
                    nca_output = models['nca'](seed, w.detach(), steps=nca_steps)
                    fake_imgs_nca = models['nca'].to_rgba(nca_output)[:, :3, :, :]
                d_fake_nca_pred, _ = models['discriminator'](fake_imgs_nca.detach(), return_features=True)
                loss_d_fake_nca = F.binary_cross_entropy_with_logits(d_fake_nca_pred, torch.zeros_like(d_fake_nca_pred))

                # Gradient Penalty
                gp_gen = gradient_penalty(models['discriminator'], real_imgs, fake_imgs_gen, DEVICE)
                gp_nca = gradient_penalty(models['discriminator'], real_imgs, fake_imgs_nca, DEVICE)
                
                loss_d = loss_d_real + (loss_d_fake_gen + loss_d_fake_nca) / 2 + (gp_gen + gp_nca) / 2 * GRADIENT_PENALTY_WEIGHT
                loss_d.backward()
                optimizers['discriminator'].step()
                metrics_history[-1]['loss_d'] = loss_d.item()

            # ---------------------------------
            # Train Generator & NCA
            # ---------------------------------
            if batch_idx % gen_train_interval == 0:
                # --- Train Generator ---
                models['generator'].zero_grad()
                noise = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
                fake_imgs_gen, w = models['generator'](noise, return_w=True)
                
                d_gen_pred, fake_features_gen = models['discriminator'](fake_imgs_gen, return_features=True)
                loss_g_adv = F.binary_cross_entropy_with_logits(d_gen_pred, torch.ones_like(d_gen_pred))
                
                with torch.no_grad():
                    _, real_features_gen = models['discriminator'](real_imgs, return_features=True)
                loss_g_perceptual = compute_perceptual_loss(real_features_gen, fake_features_gen)
                
                loss_g = loss_g_adv + loss_g_perceptual * 10.0  # Add perceptual loss with a weight
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(models['generator'].parameters(), max_norm=1.0)
                optimizers['generator'].step()
                metrics_history[-1]['loss_g'] = loss_g.item()

                # --- Train NCA ---
                models['nca'].zero_grad()
                seed = models['nca'].get_seed(BATCH_SIZE, IMG_SIZE, DEVICE)
                nca_steps = random.randint(NCA_STEPS_MIN, NCA_STEPS_MAX)
                nca_output = models['nca'](seed, w.detach(), steps=nca_steps)
                fake_imgs_nca = models['nca'].to_rgba(nca_output)[:, :3, :, :]
                
                d_nca_pred, fake_features_nca = models['discriminator'](fake_imgs_nca, return_features=True)
                loss_nca_adv = F.binary_cross_entropy_with_logits(d_nca_pred, torch.ones_like(d_nca_pred))
                
                with torch.no_grad():
                    _, real_features_nca = models['discriminator'](real_imgs, return_features=True)
                loss_nca_perceptual = compute_perceptual_loss(real_features_nca, fake_features_nca)
                
                loss_nca = loss_nca_adv + loss_nca_perceptual * 10.0 # Add perceptual loss with a weight
                loss_nca.backward()
                torch.nn.utils.clip_grad_norm_(models['nca'].parameters(), max_norm=1.0)
                optimizers['nca'].step()
                metrics_history[-1]['loss_nca'] = loss_nca.item()

            # ---------------------------------
            # Train Transformer (with cyclical modes)
            # ---------------------------------
            
            # Set transformer LR based on mode
            if transformer_mode == 'isolation':
                for g in optimizers['transformer'].param_groups:
                    g['lr'] = TRANSFORMER_LR * 0.1 # Lower LR for fine-tuning
            else:
                for g in optimizers['transformer'].param_groups:
                    g['lr'] = TRANSFORMER_LR # Normal LR

            if transformer_mode != 'dormant':
                models['transformer'].zero_grad()
                
                with torch.no_grad():
                    noise_t = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
                    fake_imgs_gen_t, w_t = models['generator'](noise_t, return_w=True)
                    seed_t = models['nca'].get_seed(BATCH_SIZE, IMG_SIZE, DEVICE)
                    nca_output_t = models['nca'](seed_t, w_t, steps=nca_steps)
                    fake_imgs_nca_t = models['nca'].to_rgba(nca_output_t)[:, :3, :, :]

                if transformer_mode == 'critic':
                    t_real = models['transformer'](real_imgs)
                    t_fake = models['transformer'](fake_imgs_gen_t.detach())
                    t_nca = models['transformer'](fake_imgs_nca_t.detach())
                    
                    loss_t_real = F.binary_cross_entropy_with_logits(t_real, torch.ones_like(t_real))
                    loss_t_fake = F.binary_cross_entropy_with_logits(t_fake, torch.zeros_like(t_fake))
                    loss_t_nca = F.binary_cross_entropy_with_logits(t_nca, torch.zeros_like(t_nca))
                    loss_t = (loss_t_real + loss_t_fake + loss_t_nca) / 3
                
                elif transformer_mode == 'isolation':
                    # Isolation mode: encourage diversity while maintaining quality
                    models['transformer'].train() # Ensure dropout is active

                    # Generate two different outputs for the same input
                    output_1, imitation_loss_1 = models['transformer'](fake_imgs_gen_t.detach(), target=real_imgs)
                    output_2, imitation_loss_2 = models['transformer'](fake_imgs_gen_t.detach(), target=real_imgs)
                    
                    # Diversity loss: rewards difference between outputs
                    # A small weight is used to not overpower the imitation loss
                    diversity_loss = -F.mse_loss(output_1, output_2)
                    DIVERSITY_WEIGHT = 0.5
                    
                    # Combine imitation and diversity losses
                    loss_t = imitation_loss_1 + imitation_loss_2 + DIVERSITY_WEIGHT * diversity_loss
                
                elif transformer_mode == 'imitator':
                    # Imitate both Generator and NCA outputs to make them look like the real image
                    gen_imitation_output, gen_imitation_loss = models['transformer'](fake_imgs_gen_t.detach(), target=real_imgs)
                    nca_imitation_output, nca_imitation_loss = models['transformer'](fake_imgs_nca_t.detach(), target=real_imgs)
                    
                    # The total loss is the average of both imitation tasks
                    loss_t = (gen_imitation_loss + nca_imitation_loss) / 2

                if epoch >= TRANSFORMER_START_EPOCH:
                    loss_t.backward()
                    optimizers['transformer'].step()

                metrics_history[-1]['loss_t'] = loss_t.item()
            else:
                # This handles the 'dormant' case
                loss_t = torch.tensor(0.0) 
                metrics_history[-1]['loss_t'] = 0.0

            # Update progress bar
            progress_bar.set_postfix({
                "D_loss": f"{loss_d.item():.3f}",
                "G_loss": f"{loss_g.item():.3f}",
                "NCA_loss": f"{loss_nca.item():.3f}",
                "T_loss": f"{loss_t.item():.3f}"
            })

            if batch_idx % 50 == 0:
                with torch.no_grad():
                    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
                    sample_gen, sample_w = models['generator'](fixed_noise, return_w=True)
                    sample_seed = models['nca'].get_seed(BATCH_SIZE, IMG_SIZE, DEVICE)
                    sample_nca_out = models['nca'](sample_seed, sample_w, steps=NCA_STEPS_MAX)
                    sample_nca = models['nca'].to_rgba(sample_nca_out)[:,:3,:,:]
                    
                    transformer_viz = torch.zeros_like(sample_gen)
                    if transformer_mode == 'imitator':
                        transformer_viz, _ = models['transformer'](sample_gen, target=real_imgs)

                    update_status(
                        f"Training epoch {epoch+1}/{EPOCHS}",
                        images=[real_imgs, sample_gen, sample_nca, transformer_viz],
                        scores={
                            'D_loss': loss_d.item(),
                            'G_loss': loss_g.item(),
                            'NCA_loss': loss_nca.item(),
                            'T_loss': loss_t.item(),
                            'transformer_mode': transformer_mode
                        }
                    )

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, models, optimizers, metrics_history)

    print("\n✅ Training finished.")
    save_checkpoint(EPOCHS, models, optimizers, metrics_history)

# Add perceptual loss calculation
def compute_perceptual_loss(real_features, fake_features, weights=None):
    if weights is None:
        weights = [1.0/len(real_features)] * len(real_features)
    
    perceptual_loss = 0
    for rf, ff, w in zip(real_features, fake_features, weights):
        # Normalize features
        rf = F.normalize(rf, p=2, dim=1)
        ff = F.normalize(ff, p=2, dim=1)
        perceptual_loss += w * F.mse_loss(ff, rf.detach())
    
    return perceptual_loss

# Add style loss calculation
def compute_gram_matrix(features):
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t) / (c * h * w)
    return gram

def compute_style_loss(real_features, fake_features):
    style_loss = 0
    for rf, ff in zip(real_features, fake_features):
        real_gram = compute_gram_matrix(rf)
        fake_gram = compute_gram_matrix(ff)
        style_loss += F.mse_loss(fake_gram, real_gram.detach())
    return style_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-training', action='store_true', help='Run training')
    args = parser.parse_args()
    
    if args.run_training:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        training_loop()
    else:
        print("Use --run-training to start training") 