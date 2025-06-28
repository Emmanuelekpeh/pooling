import os
import io
import json
import time
import base64
import random
import argparse
from threading import Thread

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

# --- Configuration ---
Z_DIM = 64  # Reduced for memory
W_DIM = 128  # Increased for richer style representation
IMG_SIZE = 64
NCA_CHANNELS = 8  # Reduced for memory
BATCH_SIZE = 1  # Further reduced batch size for stability
LR = 1e-4   
EPOCHS = 300
NCA_STEPS_MIN = 64  # Increased for better growth - needs more steps to fill 64x64 from 5x5 seed
NCA_STEPS_MAX = 96  # Increased for better growth - allows full image coverage
DEVICE = "cpu" # Force CPU
DATA_DIR = "./data/ukiyo-e-small"
SAMPLES_DIR = "./samples"
# Use environment variable for checkpoint directory with fallback to relative path
# This ensures it works with the Fly.io mounted volume at /app/checkpoints
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints" if os.path.exists("/app") else "./checkpoints")
KEEP_CHECKPOINTS = 5  # Number of recent epoch checkpoints to keep

# --- Models ---
# (Using the same simplified model definitions as before)
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, 512),  # Restored capacity for better quality  
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),    # Restored capacity for better quality
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),    # Added depth for StyleGAN quality
            nn.LeakyReLU(0.2),
            nn.Linear(512, w_dim)   # Final mapping to w space
        )
    def forward(self, x):
        return self.mapping(x)

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Style modulation layers
        self.style1 = nn.Linear(w_dim, out_channels * 2)  # Scale and bias
        self.style2 = nn.Linear(w_dim, out_channels * 2)  # Scale and bias
        
        # Noise injection - start with small random values instead of zeros
        self.noise_weight1 = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.1)
        self.noise_weight2 = nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.1)
        
    def forward(self, x, w):
        # First convolution
        x = self.conv1(x)
        
        # Add noise
        noise = torch.randn_like(x) * self.noise_weight1
        x = x + noise
        
        # Style modulation
        style = self.style1(w).unsqueeze(2).unsqueeze(3)
        scale1, bias1 = style.chunk(2, dim=1)
        x = x * (scale1 + 1) + bias1
        x = F.leaky_relu(x, 0.2)
        
        # Second convolution
        x = self.conv2(x)
        
        # Add noise
        noise = torch.randn_like(x) * self.noise_weight2
        x = x + noise
        
        # Style modulation
        style = self.style2(w).unsqueeze(2).unsqueeze(3)
        scale2, bias2 = style.chunk(2, dim=1)
        x = x * (scale2 + 1) + bias2
        x = F.leaky_relu(x, 0.2)
        
        return x

class IntegratedGenerator(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping_network = MappingNetwork(z_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, 256, 4, 4) * 0.1)  # Better initialization
        
        # Progressive upsampling with proper StyleGAN capacity
        self.gen_block1 = GeneratorBlock(256, 256, w_dim)  # 4x4 -> 8x8
        self.gen_block2 = GeneratorBlock(256, 128, w_dim)  # 8x8 -> 16x16  
        self.gen_block3 = GeneratorBlock(128, 64, w_dim)   # 16x16 -> 32x32
        self.gen_block4 = GeneratorBlock(64, 32, w_dim)    # 32x32 -> 64x64
        
        self.to_rgb = nn.Conv2d(32, 3, kernel_size=1)
        
        # Initialize to_rgb with small weights
        nn.init.xavier_uniform_(self.to_rgb.weight, gain=0.1)
        nn.init.zeros_(self.to_rgb.bias)
        
        # Initialize all conv layers properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better StyleGAN training"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)  # Increased gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear) and module != self.to_rgb:  # Don't re-init to_rgb
                nn.init.xavier_uniform_(module.weight, gain=1.0)  # Increased gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, noise, return_w=False):
        w = self.mapping_network(noise)
        x = self.const.repeat(noise.shape[0], 1, 1, 1)
        
        # Progressive generation with style modulation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 4x4 -> 8x8
        x = self.gen_block1(x, w)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 8x8 -> 16x16
        x = self.gen_block2(x, w)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 16x16 -> 32x32
        x = self.gen_block3(x, w)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 32x32 -> 64x64
        x = self.gen_block4(x, w)
        
        img = torch.tanh(self.to_rgb(x))
        if return_w:
            return img, w
        return img

class IntegratedNCA(nn.Module):
    def __init__(self, channel_n, w_dim, hidden_n=64, use_rich_conditioning=True):  # Enable by default
        super().__init__()
        self.channel_n = channel_n
        self.w_dim = w_dim
        self.use_rich_conditioning = use_rich_conditioning
        
        # LEARNABLE RUNOFF CONTROL SYSTEM
        # Network that learns to control runoff parameters based on current state
        if use_rich_conditioning:
            # When using rich conditioning, w is reduced to w_dim//3
            runoff_input_dim = channel_n + (w_dim // 3) + 4  # +4 for spatial statistics
        else:
            # When using simple conditioning, w is full w_dim
            runoff_input_dim = channel_n + w_dim + 4  # +4 for spatial statistics
            
        self.runoff_controller = nn.Sequential(
            nn.Linear(runoff_input_dim, hidden_n // 2),
            nn.ReLU(),
            nn.Linear(hidden_n // 2, hidden_n // 4),
            nn.ReLU(),
            nn.Linear(hidden_n // 4, 6),  # 6 runoff control parameters
            nn.Sigmoid()  # Output in [0,1] range for easy scaling
        )
        
        # Initialize runoff controller to encourage AGGRESSIVE exploration and survival
        with torch.no_grad():
            # Set biases to GUARANTEE survival at all costs
            self.runoff_controller[-2].bias.data = torch.tensor([
                0.76,   # exploration_rate: sigmoid(2.0) ≈ 0.88 -> MAXIMUM exploration  
                1.95,   # survival_rate: sigmoid(3.0) ≈ 0.95 -> GUARANTEED survival
                0.66,   # edge_boost: sigmoid(2.0) ≈ 0.88 -> MAXIMUM edge expansion
                0.73,   # diffusion_strength: sigmoid(1.5) ≈ 0.82 -> VERY strong diffusion
                0.05,  # spatial_threshold: sigmoid(-3.0) ≈ 0.05 -> MINIMAL threshold (almost nothing dies)
                0.79    # update_magnitude: sigmoid(1.5) ≈ 0.82 -> MAXIMUM updates
            ])
        
        if use_rich_conditioning:
            # StyleGAN-inspired conditioning: w provides semantic control, target provides details
            # Semantic features from w (30% of conditioning)
            self.w_dim_reduced = w_dim // 3  # Reduce w influence to ~30%
            self.w_projection = nn.Linear(w_dim, self.w_dim_reduced)
            
            # Rich image features (70% of conditioning) - optimized feature extractor  
            self.target_encoder = nn.Sequential(
                # Efficient multi-scale feature extraction
                nn.Conv2d(3, 16, 4, 2, 1),   # 64->32
                nn.LeakyReLU(0.2),
                nn.Conv2d(16, 32, 4, 2, 1),  # 32->16  
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),  # 16->8
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1),  # 8->4 (reduced from 128->256)
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 64),  # Reduced but still rich features (~70% of conditioning)
                nn.LeakyReLU(0.2)
            )
            
            # Conditioning fusion: small w + target features
            conditioning_dim = self.w_dim_reduced + 64  # ~40% + ~60% (optimized)
            
            # StyleGAN-inspired modulation: separate pathways for style vs content
            self.content_pathway = nn.Sequential(
                nn.Linear(64, hidden_n // 2),  # Target features -> content (reduced)
                nn.ReLU(),
                nn.Linear(hidden_n // 2, channel_n // 2)  # Half the output channels
            )
            
            self.style_pathway = nn.Sequential(
                nn.Linear(self.w_dim_reduced, hidden_n // 4),  # W features -> style (smaller)
                nn.ReLU(), 
                nn.Linear(hidden_n // 4, channel_n // 2)  # Other half of output channels
            )
            
            # Combined update network - much smaller now that we have separate pathways
            self.update_net = nn.Sequential(
                nn.Linear(channel_n * 3 + conditioning_dim, hidden_n // 2),
                nn.ReLU(),
                nn.Linear(hidden_n // 2, channel_n // 4),  # Smaller direct update
            )
        else:
            # Original simple conditioning
            conditioning_dim = w_dim
            self.update_net = nn.Sequential(
                nn.Linear(channel_n * 3 + conditioning_dim, hidden_n),
                nn.ReLU(),
                nn.Linear(hidden_n, channel_n),
            )
        
        # Initialize the last layer with larger weights for proper growth
        with torch.no_grad():
            self.update_net[-1].weight.data *= 0.5  # Increased from 0.1
        
        # Register sobel filters as buffers so they move with the model
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Add Ghost Memory + Death/Revival hybrid system after the runoff controller initialization

        # GHOST MEMORY + DEATH/REVIVAL HYBRID SYSTEM
        # Based on cellular automata collective-state computing principles
        # Memory persists through death/revival cycles, reducing need for emergency mechanisms
        
        # Ghost memory channels (persistent spatial memory)
        self.ghost_memory_channels = min(channel_n // 2, 4)  # Up to 4 ghost channels
        
        # Ghost memory encoder/decoder for compression during death/revival
        self.ghost_encoder = nn.Sequential(
            nn.Linear(channel_n, channel_n // 2),
            nn.Tanh(),  # Bounded activation for stable storage
            nn.Linear(channel_n // 2, self.ghost_memory_channels)
        )
        
        self.ghost_decoder = nn.Sequential(
            nn.Linear(self.ghost_memory_channels, channel_n // 2),
            nn.ReLU(),
            nn.Linear(channel_n // 2, channel_n)
        )
        
        # Memory diffusion kernel for spatial memory inheritance
        self.register_buffer('memory_diffusion_kernel', 
                           torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0)
        
        # Death prediction network (learns when cells should "gracefully" die)
        self.death_predictor = nn.Sequential(
            nn.Linear(channel_n + self.ghost_memory_channels + 4, hidden_n // 4),
            nn.ReLU(),
            nn.Linear(hidden_n // 4, 1),
            nn.Sigmoid()  # Probability of graceful death
        )
        
        # Revival predictor (learns when/how cells should revive with ghost memory)
        self.revival_predictor = nn.Sequential(
            nn.Linear(self.ghost_memory_channels + 4, hidden_n // 4),
            nn.ReLU(),
            nn.Linear(hidden_n // 4, 1),
            nn.Sigmoid()  # Probability of revival
        )

    def _compute_spatial_statistics(self, x):
        """Compute spatial statistics for runoff control input"""
        alive_mask = (x[:, 3:4, :, :] > 0.01).float()  # MUCH more permissive threshold
        
        # Basic statistics
        alive_count = alive_mask.sum(dim=[2, 3]) / (x.shape[2] * x.shape[3])  # Alive ratio
        
        # Edge statistics - how much of the living area is at edges
        edge_mask = self._get_edge_expansion_mask(alive_mask)
        edge_ratio = edge_mask.sum(dim=[2, 3]) / (alive_mask.sum(dim=[2, 3]) + 1e-8)
        
        # Spatial spread - measure how spread out the living cells are
        if alive_mask.sum() > 0:
            coords_y, coords_x = torch.meshgrid(
                torch.arange(x.shape[2], device=x.device, dtype=torch.float32),
                torch.arange(x.shape[3], device=x.device, dtype=torch.float32),
                indexing='ij'
            )
            coords_y = coords_y.unsqueeze(0).unsqueeze(0)
            coords_x = coords_x.unsqueeze(0).unsqueeze(0)
            
            # Weighted center of mass
            total_alive = alive_mask.sum(dim=[2, 3], keepdim=True) + 1e-8
            center_y = (alive_mask * coords_y).sum(dim=[2, 3], keepdim=True) / total_alive
            center_x = (alive_mask * coords_x).sum(dim=[2, 3], keepdim=True) / total_alive
            
            # Average distance from center (spread measure)
            dist_from_center = torch.sqrt((coords_y - center_y)**2 + (coords_x - center_x)**2)
            avg_spread = (alive_mask * dist_from_center).sum(dim=[2, 3]) / total_alive.squeeze()
            avg_spread = avg_spread / (x.shape[2] / 2)  # Normalize by half image size
        else:
            avg_spread = torch.zeros_like(alive_count)
        
        # Growth potential - areas where growth could happen
        growth_potential = self._get_edge_expansion_mask(alive_mask).sum(dim=[2, 3]) / (x.shape[2] * x.shape[3])
        
        # Stack all statistics [batch, 4]
        spatial_stats = torch.cat([
            alive_count,      # How much is alive
            edge_ratio,       # How much of alive area is at edges  
            avg_spread,       # How spread out the living cells are
            growth_potential  # How much room for growth
        ], dim=1)
        
        return spatial_stats

    def _get_adaptive_runoff_params(self, x, w):
        """Get adaptive runoff parameters based on current state"""
        # Compute spatial statistics
        spatial_stats = self._compute_spatial_statistics(x)
        
        # Global state vector: average cell state + w + spatial stats
        global_cell_state = x.mean(dim=[2, 3])  # [batch, channels]
        
        # Input to runoff controller: cell state + w + spatial stats
        runoff_input = torch.cat([global_cell_state, w, spatial_stats], dim=1)
        
        # Get adaptive parameters [batch, 6]
        raw_params = self.runoff_controller(runoff_input)
        
        # Scale parameters for ULTRA SURVIVAL - absolutely preventing NCA death
        exploration_rate = raw_params[:, 0:1] * 0.2 + 0.3     # 0.3 to 0.5 (HIGH exploration but stable)
        survival_rate = raw_params[:, 1:2] * 0.02 + 0.48      # 0.98 to 1.0 (ULTRA survival rate) 
        edge_boost = raw_params[:, 2:3] * 0.2 + 0.3           # 0.3 to 0.5 (HIGH edge expansion)
        diffusion_strength = raw_params[:, 3:4] * 0.05 + 0.2   # 0.2 to 0.25 (STRONGER spreading)
        spatial_threshold = raw_params[:, 4:5] * 0.002 + 0.0005  # 0.0005 to 0.0025 (ULTRA minimal threshold)
        update_magnitude = raw_params[:, 5:6] * 0.15 + 0.25    # 0.25 to 0.4 (STRONG but stable updates)
        
        return {
            'exploration_rate': exploration_rate,
            'survival_rate': survival_rate, 
            'edge_boost': edge_boost,
            'diffusion_strength': diffusion_strength,
            'spatial_threshold': spatial_threshold,
            'update_magnitude': update_magnitude
        }

    def _manage_ghost_memory(self, x, ghost_memory, alive_mask, spatial_stats):
        """
        Hybrid Ghost Memory + Death/Revival System
        
        This implements persistent spatial memory that survives death/revival cycles,
        based on cellular automata collective-state computing principles.
        """
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Prepare inputs for death/revival prediction
        cell_state_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, H*W, C]
        ghost_flat = ghost_memory.view(batch_size, self.ghost_memory_channels, -1).permute(0, 2, 1)  # [B, H*W, G]
        spatial_stats_expanded = spatial_stats.unsqueeze(1).expand(-1, height * width, -1)  # [B, H*W, 4]
        
        # DEATH PREDICTION: Learn when cells should gracefully die
        death_input = torch.cat([cell_state_flat, ghost_flat, spatial_stats_expanded], dim=2)
        death_probabilities = self.death_predictor(death_input).squeeze(-1)  # [B, H*W]
        death_probabilities = death_probabilities.view(batch_size, 1, height, width)
        
        # GRACEFUL DEATH: Store important information in ghost memory before dying
        dying_mask = (torch.rand_like(death_probabilities) < death_probabilities) & (alive_mask > 0.5)
        
        if dying_mask.sum() > 0:
            # Extract and encode memory from dying cells
            dying_cell_states = x * dying_mask
            dying_flat = dying_cell_states.view(batch_size, channels, -1).permute(0, 2, 1)
            
            # Encode dying cell information into compressed ghost memory
            ghost_updates = self.ghost_encoder(dying_flat)  # [B, H*W, ghost_channels]
            ghost_updates = ghost_updates.permute(0, 2, 1).view(batch_size, self.ghost_memory_channels, height, width)
            
            # Update ghost memory with exponential moving average (preserve historical information)
            memory_persistence = 0.8  # 80% of old memory persists
            memory_incorporation = 0.3  # 30% of new information incorporated
            ghost_memory = (memory_persistence * ghost_memory + 
                          memory_incorporation * ghost_updates * dying_mask.expand(-1, self.ghost_memory_channels, -1, -1))
        
        # SPATIAL MEMORY DIFFUSION: Share ghost memory between neighboring locations
        # This implements the collective-state computing principle where information is distributed
        diffused_ghost_memory = torch.zeros_like(ghost_memory)
        for ghost_channel in range(self.ghost_memory_channels):
            diffused_ghost_memory[:, ghost_channel:ghost_channel+1, :, :] = F.conv2d(
                ghost_memory[:, ghost_channel:ghost_channel+1, :, :], 
                self.memory_diffusion_kernel, 
                padding=1
            )
        ghost_memory = diffused_ghost_memory
        
        # REVIVAL PREDICTION: Learn when empty spaces should revive with ghost memory
        empty_mask = (alive_mask <= 0.01).float()
        if empty_mask.sum() > 0:
            revival_input = torch.cat([ghost_flat, spatial_stats_expanded], dim=2)
            revival_probabilities = self.revival_predictor(revival_input).squeeze(-1)  # [B, H*W]
            revival_probabilities = revival_probabilities.view(batch_size, 1, height, width)
            
            # INTELLIGENT REVIVAL: Revive cells with ghost memory inheritance
            revival_mask = (torch.rand_like(revival_probabilities) < revival_probabilities) & (empty_mask > 0.5)
            
            if revival_mask.sum() > 0:
                # Decode ghost memory for reviving cells
                reviving_ghost = ghost_memory * revival_mask.expand(-1, self.ghost_memory_channels, -1, -1)
                reviving_ghost_flat = reviving_ghost.view(batch_size, self.ghost_memory_channels, -1).permute(0, 2, 1)
                
                # Decode ghost memory into full cell state
                decoded_memory = self.ghost_decoder(reviving_ghost_flat)  # [B, H*W, channels]
                decoded_memory = decoded_memory.permute(0, 2, 1).view(batch_size, channels, height, width)
                
                # Revive cells with inherited memory (blend with random initialization)
                revival_strength = 0.6  # 60% ghost memory, 40% fresh start
                random_init = torch.randn_like(decoded_memory) * 0.1
                revival_state = revival_strength * decoded_memory + (1 - revival_strength) * random_init
                
                # Ensure reviving cells have proper alpha values
                revival_state[:, 3:4, :, :] = torch.clamp(revival_state[:, 3:4, :, :] + 0.5, 0.1, 1.0)
                
                # Apply revival to the main cell state
                x = x + revival_state * revival_mask.expand(-1, channels, -1, -1)
        
        # MEMORY DECAY: Gradually decay unused ghost memory to prevent accumulation
        memory_decay_rate = 0.995  # Very slow decay
        ghost_memory = ghost_memory * memory_decay_rate
        
        return x, ghost_memory, dying_mask, revival_mask if 'revival_mask' in locals() else torch.zeros_like(dying_mask)

    def get_seed(self, batch_size, size, device, seed_type="distributed"):
        seed = torch.zeros(batch_size, self.channel_n, size, size, device=device)
        
        if seed_type == "center":
            # Original single center seed
            center = size // 2
            radius = 2
            self._place_seed_at(seed, center, center, radius, batch_size, device)
            
        elif seed_type == "distributed":
            # Multiple seeds distributed across the grid for better spatial awareness
            # Based on Google Research isotropic NCA structured seeds approach
            num_seeds = 2  # Reduced to 2 seeds for controlled spread with enhanced runoff
            positions = self._get_distributed_positions(size, num_seeds)
            
            for pos_x, pos_y in positions:
                # Each seed gets slightly different characteristics
                radius = random.randint(1, 2)  # Vary seed size
                self._place_seed_at(seed, pos_x, pos_y, radius, batch_size, device)
                
        elif seed_type == "corners":
            # Seeds in corners + center for symmetry breaking (isotropic approach)
            corner_offset = size // 6  # Distance from actual corners
            positions = [
                (corner_offset, corner_offset),                    # Top-left
                (size - corner_offset - 1, corner_offset),        # Top-right  
                (corner_offset, size - corner_offset - 1),        # Bottom-left
                (size - corner_offset - 1, size - corner_offset - 1), # Bottom-right
                (size // 2, size // 2)                            # Center
            ]
            
            for pos_x, pos_y in positions:
                radius = 1  # Smaller seeds for corners
                self._place_seed_at(seed, pos_x, pos_y, radius, batch_size, device)
                
        elif seed_type == "line":
            # Horizontal line of seeds for testing directional growth
            y_center = size // 2
            for x in range(size // 4, 3 * size // 4, size // 8):
                self._place_seed_at(seed, x, y_center, 1, batch_size, device)
                
        return seed
    
    def _get_distributed_positions(self, size, num_seeds):
        """Generate well-distributed seed positions using grid approach"""
        positions = []
        
        # Create a grid-based distribution to ensure good coverage
        if num_seeds == 2:
            # Diagonal pattern: two seeds positioned diagonally for good coverage with enhanced runoff
            offset = size // 3
            positions = [
                (offset, offset),                      # Top-left area
                (size - offset - 1, size - offset - 1), # Bottom-right area
            ]
        elif num_seeds == 3:
            # Triangle pattern for minimal symmetry breaking
            positions = [
                (size // 4, size // 2),               # Left
                (3 * size // 4, size // 4),          # Top-right
                (3 * size // 4, 3 * size // 4)       # Bottom-right
            ]
        elif num_seeds == 7:
            # Hexagon + center pattern for maximum coverage
            center = size // 2
            radius = size // 3
            positions = [(center, center)]  # Center
            for i in range(6):
                angle = i * 60 * 3.14159 / 180  # 60-degree intervals
                x = int(center + radius * torch.cos(torch.tensor(angle)))
                y = int(center + radius * torch.sin(torch.tensor(angle)))
                x = max(2, min(size-3, x))  # Keep within bounds
                y = max(2, min(size-3, y))
                positions.append((x, y))
                
        return positions
    
    def _place_seed_at(self, seed, center_x, center_y, radius, batch_size, device):
        """Place a seed at specific coordinates with given radius"""
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                x, y = center_x + i, center_y + j
                if 0 <= x < seed.shape[2] and 0 <= y < seed.shape[3]:
                    # Set RGB values with variation based on position for diversity
                    position_hash = (x * 1000 + y) % 1000 / 1000.0
                    
                    seed[:, 0, x, y] = 0.5 + (0.3 + position_hash * 0.2) * torch.rand(batch_size, device=device)  # R stronger initial
                    seed[:, 1, x, y] = 0.4 + (0.3 + position_hash * 0.3) * torch.rand(batch_size, device=device)  # G stronger initial
                    seed[:, 2, x, y] = 0.7 + (0.2 + position_hash * 0.1) * torch.rand(batch_size, device=device)  # B stronger initial
                    seed[:, 3, x, y] = 3.0  # Alpha (alive) - ULTRA MAXIMUM to guarantee seed survival (increased from 2.0)
                    
                    # Add diverse hidden channel values for different seed behaviors
                    if self.channel_n > 4:
                        # Each seed location gets unique hidden state based on position
                        hidden_seed = torch.randn(batch_size, self.channel_n - 4, device=device) * 0.3
                        # Add position-dependent bias to encourage different behaviors
                        position_bias = torch.tensor([position_hash * 2 - 1] * (self.channel_n - 4), device=device).unsqueeze(0)
                        seed[:, 4:, x, y] = hidden_seed + position_bias * 0.2

    def _get_edge_expansion_mask(self, alive_mask):
        """Identify edge cells that are good candidates for spatial expansion"""
        # Find cells that are alive
        alive = (alive_mask > 0.01).float()  # MUCH more permissive threshold
        
        # Find cells that are at the edge of living regions (have empty neighbors)
        # Use erosion to find internal cells, subtract from alive to get edge cells
        eroded = F.max_pool2d(-alive, kernel_size=3, stride=1, padding=1)
        eroded = (-eroded).clamp(0, 1)  # Convert back to positive
        edge_cells = alive - eroded  # Cells that are alive but became "dead" after erosion = edge cells
        
        # Find empty cells that are adjacent to living cells (expansion candidates)
        neighbor_alive = F.max_pool2d(alive, kernel_size=3, stride=1, padding=1)
        expansion_candidates = (neighbor_alive > 0.01).float() * (1 - alive)  # Empty cells with living neighbors
        
        # Combine edge cells and expansion candidates for full exploration mask
        exploration_opportunities = edge_cells + expansion_candidates * 0.6  # Weight expansion slightly less
        
        return exploration_opportunities.clamp(0, 1)

    def to_rgba(self, x):
        # Clamp RGB values to [-1, 1] range and ensure alpha is positive  
        rgba = x[:, :4, :, :].clone()
        rgb_clamped = torch.tanh(rgba[:, :3, :, :])  # RGB in [-1, 1]
        
        # FIXED: Don't use sigmoid on alpha - it makes everything alive!
        # Instead, just clamp alpha to [0, 1] and preserve zeros - non-in-place
        alpha_clamped = torch.clamp(rgba[:, 3:4, :, :], 0, 1)  # Alpha in [0, 1] but preserves zeros
        
        # Combine into final RGBA tensor without in-place operations
        rgba_final = torch.cat([rgb_clamped, alpha_clamped], dim=1)
        return rgba_final

    def perceive(self, x):
        # Use repeat with correct dimensions - sobel filters are [1, 1, 3, 3], need [channel_n, 1, 3, 3]
        sobel_x_expanded = self.sobel_x.repeat(self.channel_n, 1, 1, 1)
        sobel_y_expanded = self.sobel_y.repeat(self.channel_n, 1, 1, 1)
        
        grad_x = F.conv2d(x, sobel_x_expanded, padding=1, groups=self.channel_n)
        grad_y = F.conv2d(x, sobel_y_expanded, padding=1, groups=self.channel_n)
        return torch.cat((x, grad_x, grad_y), 1)

    def forward(self, x, w, steps, target_img=None):
        if self.use_rich_conditioning:
            # Rich conditioning REQUIRES target_img - error if not provided
            if target_img is None:
                raise ValueError("Rich conditioning is enabled but target_img is None. Target image is required for rich conditioning.")
            
            # Initialize Ghost Memory for death/revival persistence
            batch_size, channels, height, width = x.shape
            ghost_memory = torch.zeros(batch_size, self.ghost_memory_channels, height, width, 
                                     device=x.device, dtype=x.dtype)
            
            # StyleGAN-inspired approach: separate semantic and content processing
            # 1. Extract rich target features (60% influence)  
            target_features = self.target_encoder(target_img)  # [batch, 64]
            
            # 2. Reduce w vector to semantic essentials (40% influence)
            w_semantic = self.w_projection(w)  # [batch, w_dim//3]
            
            # 3. Create separate content and style updates for each step
            for step in range(steps):
                alive_mask = (x[:, 3:4, :, :] > 0.01).float()  # MUCH more permissive threshold
                
                # Perception (unchanged)
                perceived = self.perceive(x)
                perceived = perceived.permute(0, 2, 3, 1).reshape(-1, self.channel_n * 3)
                
                # CONTENT PATHWAY: Target features drive detailed growth
                content_updates = self.content_pathway(target_features)  # [batch, channel_n//2]
                # Ensure content_updates is 2D [batch, features] before expanding
                if content_updates.dim() == 1:
                    content_updates = content_updates.unsqueeze(0)
                content_updates_expanded = content_updates.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.shape[2], x.shape[3])
                content_updates_reshaped = content_updates_expanded.permute(0, 2, 3, 1).reshape(-1, content_updates.shape[1])
                
                # STYLE PATHWAY: W features drive semantic guidance  
                style_updates = self.style_pathway(w_semantic)  # [batch, channel_n//2]
                # Ensure style_updates is 2D [batch, features] before expanding
                if style_updates.dim() == 1:
                    style_updates = style_updates.unsqueeze(0)
                style_updates_expanded = style_updates.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.shape[2], x.shape[3])
                style_updates_reshaped = style_updates_expanded.permute(0, 2, 3, 1).reshape(-1, style_updates.shape[1])
                
                # DIRECT PATHWAY: Small direct updates based on perception + combined conditioning
                combined_conditioning = torch.cat([w_semantic, target_features], dim=1)
                # Ensure combined_conditioning is 2D [batch, features] before expanding
                if combined_conditioning.dim() == 1:
                    combined_conditioning = combined_conditioning.unsqueeze(0)
                conditioning_expanded = combined_conditioning.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.shape[2], x.shape[3])
                conditioning_reshaped = conditioning_expanded.permute(0, 2, 3, 1).reshape(-1, combined_conditioning.shape[1])
                
                update_input = torch.cat([perceived, conditioning_reshaped], dim=1)
                direct_updates = self.update_net(update_input)  # [N, channel_n//4]
                
                # COMBINE ALL PATHWAYS: Content (50%) + Style (25%) + Direct (25%)
                # Pad direct updates to match full channel count
                total_points = perceived.shape[0]
                combined_updates = torch.zeros(total_points, self.channel_n, device=x.device)
                
                # Content pathway gets first half of channels (dominant)
                combined_updates[:, :self.channel_n//2] = content_updates_reshaped * 0.7  # 70% influence
                
                # Style pathway gets remaining half of channels (semantic guidance)
                combined_updates[:, self.channel_n//2:] = style_updates_reshaped * 0.3  # 30% influence
                
                # Direct updates get added to first quarter (fine-tuning)
                quarter_channels = self.channel_n // 4
                combined_updates[:, :quarter_channels] += direct_updates * 0.2  # Small direct adjustment
                
                # Reshape back to spatial format
                ds = combined_updates.reshape(x.shape[0], x.shape[2], x.shape[3], self.channel_n).permute(0, 3, 1, 2)
                
                # ADAPTIVE RUNOFF CONTROL - Let the system learn its own exploration behavior
                runoff_params = self._get_adaptive_runoff_params(x, w_semantic)
                
                # GHOST MEMORY + DEATH/REVIVAL HYBRID SYSTEM
                # This handles intelligent death/revival with persistent memory
                spatial_stats = self._compute_spatial_statistics(x)
                x, ghost_memory, dying_mask, revival_mask = self._manage_ghost_memory(x, ghost_memory, alive_mask, spatial_stats)
                
                # Update alive mask after ghost memory operations
                alive_mask = (x[:, 3:4, :, :] > 0.01).float()
                
                # Adaptive exploration based on learned parameters (enhanced with ghost memory info)
                exploration_bonus = torch.rand_like(alive_mask) * runoff_params['exploration_rate'].unsqueeze(-1).unsqueeze(-1)
                stochastic_mask = (torch.rand_like(alive_mask) < runoff_params['survival_rate'].unsqueeze(-1).unsqueeze(-1)).float()
                
                # Adaptive edge expansion
                edge_mask = self._get_edge_expansion_mask(alive_mask)
                exploration_mask = alive_mask * stochastic_mask + edge_mask * runoff_params['edge_boost'].unsqueeze(-1).unsqueeze(-1)
                
                # Apply updates with adaptive magnitude
                update_mag = runoff_params['update_magnitude'].unsqueeze(-1).unsqueeze(-1)
                x = x + ds * exploration_mask * update_mag + ds * exploration_bonus * (update_mag * 0.5)
                
                # Adaptive life dynamics - FIXED: Ensure life_mask is never all zeros
                neighbor_life = F.max_pool2d(alive_mask, kernel_size=3, stride=1, padding=2)
                life_threshold = runoff_params['spatial_threshold'].unsqueeze(-1).unsqueeze(-1)
                life_mask = (neighbor_life > life_threshold).float()
                
                # INTELLIGENT RESURRECTION (enhanced by ghost memory system)
                # Keep most alive cells alive, but rely more on ghost memory for revivals
                life_mask = torch.maximum(life_mask, alive_mask * 0.75)  # 95% survival (reduced from 99%)
                # Moderate base survival (rely more on ghost memory revival system)
                life_mask = torch.maximum(life_mask, torch.ones_like(life_mask) * 0.2)  # 20% base survival (reduced from 40%)
                
                # MINIMAL EMERGENCY RESURRECTION (since ghost memory handles most cases)
                total_alive = alive_mask.sum()
                if total_alive < (alive_mask.numel() * 0.005):  # Only emergency at < 0.5% alive (much lower threshold)
                    # Only print occasionally to reduce spam
                    if not hasattr(self, '_resurrection_counter'):
                        self._resurrection_counter = 0
                    self._resurrection_counter += 1
                    if self._resurrection_counter % 20 == 1:  # Even less frequent printing
                        print(f"MINIMAL EMERGENCY: Only {total_alive} cells alive, ghost memory should handle this (occurrence #{self._resurrection_counter})")
                    
                    # Much gentler emergency resurrection (ghost memory does the heavy lifting)
                    resurrection_mask = torch.rand_like(life_mask) < 0.05  # Reduced from 15% to 5%
                    life_mask = torch.maximum(life_mask, resurrection_mask.float())
                    
                    # Fewer guaranteed spots (ghost memory provides spatial coherence)
                    h, w = life_mask.shape[2], life_mask.shape[3]
                    living_spots = 3  # Reduced from 8 to 3
                    for _ in range(living_spots):
                        spot_x, spot_y = torch.randint(0, h, (1,)).item(), torch.randint(0, w, (1,)).item()
                        # Smaller 3x3 regions (reduced from 5x5)
                        life_mask[:, :, max(0, spot_x-1):min(h, spot_x+2), max(0, spot_y-1):min(w, spot_y+2)] = 1.0
                
                # Adaptive spatial diffusion
                diffusion_kernel = torch.ones(1, 1, 7, 7, device=x.device) / 49.0
                alpha_diffused = F.conv2d(x[:, 3:4, :, :], diffusion_kernel, padding=3)
                diffusion_strength = runoff_params['diffusion_strength'].unsqueeze(-1).unsqueeze(-1)
                spatial_boost = (alpha_diffused > diffusion_strength).float() * diffusion_strength
                
                x = x * life_mask
                # Create new alpha channel instead of in-place modification to avoid gradient issues
                new_alpha = torch.clamp(x[:, 3:4, :, :] + 0.02 * alive_mask + spatial_boost, 0, 1.3)
                
                # ALPHA EMERGENCY BOOST: If alpha is too low everywhere, force some high values
                alpha_mean = new_alpha.mean()
                if alpha_mean < 0.05:  # Very low alpha
                    # Only print occasionally to reduce spam (every 20th occurrence)
                    if not hasattr(self, '_alpha_boost_counter'):
                        self._alpha_boost_counter = 0
                    self._alpha_boost_counter += 1
                    if self._alpha_boost_counter % 20 == 1:
                        print(f"ALPHA EMERGENCY BOOST: Mean alpha {alpha_mean:.4f} (occurrence #{self._alpha_boost_counter})")
                    
                    # MUCH MORE AGGRESSIVE emergency boost
                    emergency_alpha = torch.rand_like(new_alpha) * 0.1  # Increased from 0.1 to 0.3
                    high_boost_mask = torch.rand_like(new_alpha) < 0.08  # Increased from 2% to 8% of cells
                    emergency_alpha[high_boost_mask] = 0.2  # Increased from 0.8 to 1.2
                    
                    # Additional: Create guaranteed survival spots
                    h, w = new_alpha.shape[2], new_alpha.shape[3]
                    survival_spots = 5  # Number of guaranteed spots
                    for _ in range(survival_spots):
                        spot_x, spot_y = torch.randint(0, h, (1,)).item(), torch.randint(0, w, (1,)).item()
                        emergency_alpha[:, :, max(0, spot_x-2):min(h, spot_x+3), max(0, spot_y-2):min(w, spot_y+3)] = 1.0
                    
                    new_alpha = torch.clamp(new_alpha + emergency_alpha, 0, 2.0)  # Increased max from 1.5 to 2.0
                
                x = torch.cat([x[:, :3, :, :], new_alpha, x[:, 4:, :, :]], dim=1)
                
                # Memory cleanup
                if step % 20 == 0:
                    del perceived, content_updates_expanded, style_updates_expanded, conditioning_expanded
                    del content_updates_reshaped, style_updates_reshaped, conditioning_reshaped, update_input, direct_updates, combined_updates, ds
                    del stochastic_mask, edge_mask, exploration_bonus, exploration_mask, neighbor_life, life_mask, alpha_diffused, spatial_boost
            
            # Return the final NCA state after all steps
            return x
        
        else:
            # Simple conditioning path (fallback)
            for step in range(steps):
                alive_mask = (x[:, 3:4, :, :] > 0.01).float()
                
                # Perception
                perceived = self.perceive(x)
                perceived = perceived.permute(0, 2, 3, 1).reshape(-1, self.channel_n * 3)
                
                # Direct conditioning
                w_expanded = w.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.shape[2], x.shape[3])
                w_reshaped = w_expanded.permute(0, 2, 3, 1).reshape(-1, w.shape[1])
                
                update_input = torch.cat([perceived, w_reshaped], dim=1)
                ds = self.update_net(update_input)
                ds = ds.reshape(x.shape[0], x.shape[2], x.shape[3], self.channel_n).permute(0, 3, 1, 2)
                
                # Apply updates
                x = x + ds * alive_mask
                
                # Basic life dynamics
                neighbor_life = F.max_pool2d(alive_mask, kernel_size=3, stride=1, padding=1)
                life_mask = (neighbor_life > 0.001).float()
                life_mask = torch.maximum(life_mask, alive_mask * 0.7)
                x = x * life_mask
            
            return x

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64 -> 32, increased capacity
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), # 32 -> 16, increased capacity
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # 16 -> 8, increased capacity
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), # 8 -> 4, increased capacity
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0), # 4 -> 1 (final layer)
        )
    def forward(self, img):
        return self.model(img).view(-1)

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

# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # Normalize path separators for Windows compatibility
        img_path = os.path.normpath(img_path)
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

# --- Checkpoint Management ---
def save_checkpoint(epoch, models, optimizers, losses, scores, checkpoint_dir, keep_last_n=5):
    """Save training checkpoint with automatic cleanup"""
    try:
        print(f"Attempting to save checkpoint to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True, mode=0o777)  # Ensure directory has write permissions
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': models['generator'].state_dict(),
            'nca_state_dict': models['nca'].state_dict(),
            'discriminator_state_dict': models['discriminator'].state_dict(),
            'gen_evaluator_state_dict': models['gen_evaluator'].state_dict(),
            'nca_evaluator_state_dict': models['nca_evaluator'].state_dict(),
            'transformer_critic_state_dict': models['transformer_critic'].state_dict(),
            'transformer_mode': models['transformer_critic'].mode,
            'cross_learning_system_state_dict': models['cross_learning_system'].state_dict(),
            'opt_gen_state_dict': optimizers['gen'].state_dict(),
            'opt_nca_state_dict': optimizers['nca'].state_dict(),
            'opt_disc_state_dict': optimizers['disc'].state_dict(),
            'opt_gen_eval_state_dict': optimizers['gen_eval'].state_dict(),
            'opt_nca_eval_state_dict': optimizers['nca_eval'].state_dict(),
            'opt_transformer_state_dict': optimizers['transformer'].state_dict(),
            'opt_cross_learning_state_dict': optimizers['cross_learning'].state_dict(),
            'losses': losses,
            'scores': scores
        }
        
        # Save epoch-specific checkpoint
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)
        print(f"Saved checkpoint to {epoch_path}")
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")
        
        # Cleanup old checkpoints - keep only the last N epoch checkpoints
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
        
        # Verify files exist
        if os.path.exists(epoch_path) and os.path.exists(latest_path):
            print(f"Checkpoint saved at epoch {epoch} - Verified files exist")
            print(f"Epoch checkpoint size: {os.path.getsize(epoch_path)} bytes")
            print(f"Latest checkpoint size: {os.path.getsize(latest_path)} bytes")
        else:
            print(f"Warning: Checkpoint files not found after saving")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        # Try to create a simple test file to check permissions
        try:
            test_file = os.path.join(checkpoint_dir, "test_write.txt")
            with open(test_file, "w") as f:
                f.write("Test write access")
            print(f"Successfully created test file at {test_file}")
            os.remove(test_file)
        except Exception as test_e:
            print(f"Error creating test file: {str(test_e)}")

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

def load_checkpoint(checkpoint_path, models, optimizers):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found, starting from scratch")
        return 0, [], []
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    models['generator'].load_state_dict(checkpoint['generator_state_dict'])
    models['nca'].load_state_dict(checkpoint['nca_state_dict'])
    models['discriminator'].load_state_dict(checkpoint['discriminator_state_dict'])
    models['gen_evaluator'].load_state_dict(checkpoint['gen_evaluator_state_dict'])
    models['nca_evaluator'].load_state_dict(checkpoint['nca_evaluator_state_dict'])
    models['transformer_critic'].load_state_dict(checkpoint['transformer_critic_state_dict'])
    models['transformer_critic'].set_mode(checkpoint['transformer_mode'])
    models['cross_learning_system'].load_state_dict(checkpoint['cross_learning_system_state_dict'])
    
    optimizers['gen'].load_state_dict(checkpoint['opt_gen_state_dict'])
    optimizers['nca'].load_state_dict(checkpoint['opt_nca_state_dict'])
    optimizers['disc'].load_state_dict(checkpoint['opt_disc_state_dict'])
    optimizers['gen_eval'].load_state_dict(checkpoint['opt_gen_eval_state_dict'])
    optimizers['nca_eval'].load_state_dict(checkpoint['opt_nca_eval_state_dict'])
    optimizers['transformer'].load_state_dict(checkpoint['opt_transformer_state_dict'])
    optimizers['cross_learning'].load_state_dict(checkpoint['opt_cross_learning_state_dict'])
    
    epoch = checkpoint['epoch']
    losses = checkpoint.get('losses', [])
    scores = checkpoint.get('scores', [])
    
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, losses, scores

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NCA vs StyleGAN Training</title>
    <style>
        body { font-family: sans-serif; background-color: #282c34; color: #abb2bf; text-align: center; }
        .container { display: flex; justify-content: center; align-items: flex-start; gap: 40px; margin-top: 20px; flex-wrap: wrap; }
        @media (max-width: 2500px) { .container { flex-direction: column; align-items: center; } }
        .column { display: flex; flex-direction: column; align-items: center; }
        h1, h2 { color: #61afef; }
        img { border: 2px solid #61afef; width: 800px; height: 800px; image-rendering: pixelated; object-fit: contain; max-width: none !important; max-height: none !important; }
        #status { margin-top: 20px; font-size: 1.2em; min-height: 50px; padding: 10px; border-radius: 5px; background-color: #3b4048; }
        .error { color: #e06c75; border: 1px solid #e06c75; }
        .scores { margin-top: 10px; font-size: 0.9em; color: #98c379; }
    </style>
</head>
<body>
    <h1>NCA vs StyleGAN Training Status</h1>
    <div id="status">Connecting...</div>
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
    </div>
    <script>
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
                        }
                        
                        if (data.scores) {
                            document.getElementById('gen-scores').textContent = `Quality Score: ${data.scores.gen_quality?.toFixed(3) || '--'}`;
                            document.getElementById('nca-scores').textContent = `Quality Score: ${data.scores.nca_quality?.toFixed(3) || '--'}`;
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
    """Converts the first tensor in a batch to a large base64 encoded PNG image."""
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
        # Create PIL Image and resize to make it much larger
        im = Image.fromarray(ndarr)
        # Resize with nearest neighbor for crisp pixels
        im = im.resize((512, 512), Image.NEAREST)
        # Save to buffer
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error in tensor_to_b64: {str(e)}")
        # Create a solid black image as fallback
        black_img = Image.new('RGB', (512, 512), color='black')
        buffer = io.BytesIO()
        black_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def update_status(status_message, images=None, scores=None, error=False):
    """Writes the current status, images, and scores to a JSON file."""
    try:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        # The 'images' argument is expected to be a list of tensors
        b64_images = []
        if images:
            for i, img_tensor in enumerate(images):
                try:
                    b64_img = tensor_to_b64(img_tensor)
                    b64_images.append(b64_img)
                except Exception as e:
                    print(f"Error processing image {i}: {str(e)}")
                    # Add a black image as fallback
                    black_img = Image.new('RGB', (512, 512), color='black')
                    buffer = io.BytesIO()
                    black_img.save(buffer, format="PNG")
                    b64_images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        
        # Ensure we always have exactly 3 images for the web interface
        while len(b64_images) < 3:
            black_img = Image.new('RGB', (512, 512), color='black')
            buffer = io.BytesIO()
            black_img.save(buffer, format="PNG")
            b64_images.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        
        with open(STATUS_FILE, 'w') as f:
            json.dump({
                'status': status_message,
                'images': b64_images,
                'scores': scores or {},
                'error': error,
                'timestamp': time.time()
            }, f)
    except Exception as e:
        print(f"Error in update_status: {str(e)}")
        # Write minimal status if everything fails
        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump({
                    'status': f"Error: {str(e)}",
                    'images': [],
                    'scores': {},
                    'error': True,
                    'timestamp': time.time()
                }, f)
        except:
            pass  # If even this fails, give up

# --- Training Function (to be run by the 'worker' process) ---
def training_loop():
    """The main training loop with persistence and mutual evaluation."""
    try:
        # Enable anomaly detection to find gradient issues
        torch.autograd.set_detect_anomaly(True)
        update_status("Initializing models and data...")
        
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
        
        # --- Create separate models and optimizers ---
        generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
        nca_model = IntegratedNCA(channel_n=NCA_CHANNELS, w_dim=W_DIM, use_rich_conditioning=True).to(DEVICE)  # Enable StyleGAN-inspired conditioning
        discriminator = Discriminator(IMG_SIZE).to(DEVICE)
        
        # Cross-evaluation networks
        gen_evaluator = CrossEvaluator(IMG_SIZE).to(DEVICE)  # Evaluates NCA outputs
        nca_evaluator = CrossEvaluator(IMG_SIZE).to(DEVICE)  # Evaluates Generator outputs
        
        # Transformer Critic/Imitator - evolves at epoch 350
        transformer_critic = TransformerCritic(img_size=IMG_SIZE, embed_dim=128, num_heads=4, 
                                             num_layers=3, dropout=0.1).to(DEVICE)
        
        # Enhanced Cross-Learning System with Signal Weighting
        cross_learning_system = EnhancedCrossLearningSystem(img_size=IMG_SIZE, num_models=5).to(DEVICE)

        models = {
            'generator': generator,
            'nca': nca_model,
            'discriminator': discriminator,
            'gen_evaluator': gen_evaluator,
            'nca_evaluator': nca_evaluator,
            'transformer_critic': transformer_critic,
            'cross_learning_system': cross_learning_system
        }

        # Optimizers
        opt_gen = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))  # Fixed: Train ALL generator parameters
        opt_nca = optim.Adam(nca_model.parameters(), lr=LR * 10, betas=(0.5, 0.999))
        opt_disc = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
        opt_gen_eval = optim.Adam(gen_evaluator.parameters(), lr=LR, betas=(0.5, 0.999))
        opt_nca_eval = optim.Adam(nca_evaluator.parameters(), lr=LR, betas=(0.5, 0.999))
        opt_transformer = optim.Adam(transformer_critic.parameters(), lr=LR * 0.5, betas=(0.5, 0.999))  # Slower LR for transformer
        opt_cross_learning = optim.Adam(cross_learning_system.parameters(), lr=LR * 0.8, betas=(0.5, 0.999))  # Cross-learning optimizer

        optimizers = {
            'gen': opt_gen,
            'nca': opt_nca,
            'disc': opt_disc,
            'gen_eval': opt_gen_eval,
            'nca_eval': opt_nca_eval,
            'transformer': opt_transformer,
            'cross_learning': opt_cross_learning
        }
        
        # Historical averaging for stability (BigGAN technique)
        historical_params = {
            'gen': None,
            'nca': None
        }
        historical_decay = 0.999  # Exponential moving average decay
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
        start_epoch, loss_history, score_history = load_checkpoint(checkpoint_path, models, optimizers)
        
        # Initialize metrics tracking for mini-graphs
        metrics_history = []
        
        dataset = ImageDataset(DATA_DIR, img_size=IMG_SIZE)
        print(f"Found {len(dataset)} images in dataset")
        if len(dataset) == 0:
            raise ValueError(f"No images found in {DATA_DIR}")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

        fixed_noise = torch.randn(BATCH_SIZE, Z_DIM).to(DEVICE)
        
        update_status(f"Resuming training from epoch {start_epoch + 1}...")
        
        print(f"Training configuration:")
        print(f"- IMG_SIZE: {IMG_SIZE}")
        print(f"- BATCH_SIZE: {BATCH_SIZE}")
        print(f"- DEVICE: {DEVICE}")
        print(f"- Dataset size: {len(dataset)}")
        print(f"- Starting from epoch: {start_epoch + 1}")
        
        for epoch in range(start_epoch, EPOCHS):
            epoch_scores = {'gen_quality': 0.0, 'nca_quality': 0.0, 'gen_eval_loss': 0.0, 'nca_eval_loss': 0.0, 
                           'transformer_loss': 0.0, 'transformer_mode': 'critic'}
            batch_count = 0
            
            # TRANSFORMER MODE SWITCHING - Critical transition at epoch 350
            if epoch >= 350 and transformer_critic.mode == "critic":
                print(f"\n🔄 EPOCH {epoch}: TRANSFORMER EVOLUTION - Critic → Imitator")
                transformer_critic.set_mode("imitator")
                # Reduce learning rate for fine-tuning in imitator mode
                for param_group in optimizers['transformer'].param_groups:
                    param_group['lr'] *= 0.1
                update_status(f"Transformer evolved to imitator mode at epoch {epoch}")
            
            epoch_scores['transformer_mode'] = transformer_critic.mode
            
            progress_bar = tqdm(dataloader, desc=f"E:{epoch+1}", leave=False)
            for batch_idx, real_imgs in enumerate(progress_bar):
                real_imgs = real_imgs.to(DEVICE)
                batch_count += 1
                
                # Generate w vector from noise
                noise = torch.randn(real_imgs.shape[0], Z_DIM).to(DEVICE)
                
                # --- Train Discriminator ---
                discriminator.zero_grad()
                
                # On real images
                disc_real_pred = discriminator(real_imgs)
                loss_disc_real = F.binary_cross_entropy_with_logits(disc_real_pred, torch.ones_like(disc_real_pred))
                
                # On fake images from Generator
                with torch.no_grad():
                    fake_imgs_gen, w = generator(noise, return_w=True)
                disc_fake_gen_pred = discriminator(fake_imgs_gen.detach())
                loss_disc_fake_gen = F.binary_cross_entropy_with_logits(disc_fake_gen_pred, torch.zeros_like(disc_fake_gen_pred))

                # On fake images from NCA  
                with torch.no_grad():
                    seed = nca_model.get_seed(batch_size=real_imgs.shape[0], size=IMG_SIZE, device=DEVICE, seed_type="distributed")
                    nca_steps = random.randint(NCA_STEPS_MIN, NCA_STEPS_MAX)
                    nca_output_grid = nca_model(seed, w.detach(), steps=nca_steps, target_img=real_imgs)  # Pass target images for rich conditioning
                    nca_rgba = nca_model.to_rgba(nca_output_grid)
                    fake_imgs_nca = nca_rgba[:, :3, :, :]  # Already in [-1, 1] range from tanh
                disc_fake_nca_pred = discriminator(fake_imgs_nca.detach())
                loss_disc_fake_nca = F.binary_cross_entropy_with_logits(disc_fake_nca_pred, torch.zeros_like(disc_fake_nca_pred))

                loss_disc = (loss_disc_real + loss_disc_fake_gen + loss_disc_fake_nca) / 3
                loss_disc.backward()
                opt_disc.step()

                # --- Train Cross-Evaluators ---
                gen_evaluator.zero_grad()
                nca_evaluator.zero_grad()
                
                # Train Generator's evaluator on NCA outputs (teach it to score NCA quality)
                nca_quality_pred = gen_evaluator(fake_imgs_nca.detach())
                # Use discriminator's opinion as ground truth for quality
                with torch.no_grad():
                    nca_quality_target = torch.sigmoid(disc_fake_nca_pred).detach()
                    # Ensure both tensors have the same shape
                    if nca_quality_pred.dim() != nca_quality_target.dim():
                        nca_quality_target = nca_quality_target.view_as(nca_quality_pred)
                loss_gen_eval = F.mse_loss(nca_quality_pred, nca_quality_target)
                loss_gen_eval.backward()
                opt_gen_eval.step()
                
                # Train NCA's evaluator on Generator outputs
                gen_quality_pred = nca_evaluator(fake_imgs_gen.detach())
                with torch.no_grad():
                    gen_quality_target = torch.sigmoid(disc_fake_gen_pred).detach()
                    # Ensure both tensors have the same shape
                    if gen_quality_pred.dim() != gen_quality_target.dim():
                        gen_quality_target = gen_quality_target.view_as(gen_quality_pred)
                loss_nca_eval = F.mse_loss(gen_quality_pred, gen_quality_target)
                loss_nca_eval.backward()
                opt_nca_eval.step()

                # --- Train Transformer Critic/Imitator ---
                transformer_critic.zero_grad()
                
                if transformer_critic.mode == "critic":
                    # Critic mode: Learn to evaluate image quality
                    # Train on both generated images and real images
                    real_transformer_score = transformer_critic(real_imgs)
                    gen_transformer_score = transformer_critic(fake_imgs_gen.detach())
                    nca_transformer_score = transformer_critic(fake_imgs_nca.detach())
                    
                    # Transformer learns to distinguish quality (real=1, fake=lower scores)
                    real_target = torch.ones_like(real_transformer_score) * 0.9  # High quality
                    gen_target = torch.sigmoid(disc_fake_gen_pred).detach()  # Use discriminator as teacher
                    nca_target = torch.sigmoid(disc_fake_nca_pred).detach()  # Use discriminator as teacher
                    
                    loss_transformer_real = F.mse_loss(real_transformer_score, real_target)
                    loss_transformer_gen = F.mse_loss(gen_transformer_score, gen_target)
                    loss_transformer_nca = F.mse_loss(nca_transformer_score, nca_target)
                    
                    loss_transformer = (loss_transformer_real + loss_transformer_gen + loss_transformer_nca) / 3
                    
                elif transformer_critic.mode == "imitator":
                    # Imitator mode: Learn to generate images that match targets
                    # Use both generator and NCA outputs as input, real images as targets
                    
                    # Imitate generator outputs
                    gen_imitation, gen_imitation_loss = transformer_critic(fake_imgs_gen.detach(), real_imgs)
                    
                    # Imitate NCA outputs  
                    nca_imitation, nca_imitation_loss = transformer_critic(fake_imgs_nca.detach(), real_imgs)
                    
                    # Combined imitation loss
                    loss_transformer = (gen_imitation_loss + nca_imitation_loss) / 2
                    
                    # Additional perceptual loss using discriminator features
                    with torch.no_grad():
                        real_features = discriminator.model[:-1](real_imgs)
                        gen_imitation_features = discriminator.model[:-1](gen_imitation)
                        nca_imitation_features = discriminator.model[:-1](nca_imitation)
                    
                    perceptual_loss = (F.mse_loss(gen_imitation_features, real_features) + 
                                     F.mse_loss(nca_imitation_features, real_features)) / 2
                    
                    loss_transformer += 0.1 * perceptual_loss  # Small perceptual weight
                
                loss_transformer.backward()
                opt_transformer.step()

                # --- Enhanced Cross-Learning with Signal Weighting ---
                cross_learning_system.zero_grad()
                
                # Prepare images for cross-learning analysis
                images_dict = {
                    'real': real_imgs,
                    'generator': fake_imgs_gen.detach(),
                    'nca': fake_imgs_nca.detach(),
                    'discriminator': real_imgs  # Use real images as discriminator reference
                }
                
                # Add transformer output if available
                if transformer_critic.mode == "imitator":
                    with torch.no_grad():
                        transformer_gen_output, _ = transformer_critic(fake_imgs_gen.detach(), real_imgs)
                        transformer_nca_output, _ = transformer_critic(fake_imgs_nca.detach(), real_imgs)
                        images_dict['transformer'] = transformer_gen_output.detach()
                
                # Calculate progress metrics
                epoch_progress = batch_idx / len(dataloader)
                batch_progress = (epoch * len(dataloader) + batch_idx) / (EPOCHS * len(dataloader))
                
                # Get ground truth performance (discriminator scores as proxy)
                with torch.no_grad():
                    disc_real_score = torch.sigmoid(discriminator(real_imgs))
                    disc_gen_score = torch.sigmoid(disc_fake_gen_pred)
                    disc_nca_score = torch.sigmoid(disc_fake_nca_pred)
                    target_performance = (disc_real_score + disc_gen_score + disc_nca_score) / 3
                
                # Run enhanced cross-learning
                cross_learning_results = cross_learning_system(
                    images_dict, epoch_progress, batch_progress, target_performance
                )
                
                # Extract weighted signals
                weighted_gen_quality = cross_learning_results['weighted_gen_quality']
                weighted_nca_quality = cross_learning_results['weighted_nca_quality']
                ensemble_prediction = cross_learning_results['ensemble_prediction']
                
                # Cross-learning loss (teach the system to predict quality accurately)
                # Ensure both tensors have compatible shapes
                ensemble_pred_scalar = ensemble_prediction.mean() if ensemble_prediction.numel() > 1 else ensemble_prediction
                target_scalar = target_performance.squeeze()
                cross_learning_loss = F.mse_loss(ensemble_pred_scalar, target_scalar)
                cross_learning_loss.backward()
                opt_cross_learning.step()

                # --- Train Generator and NCA with Enhanced Mutual Evaluation ---
                generator.zero_grad()
                nca_model.zero_grad()

                # Re-generate images and get discriminator feedback
                fake_imgs_gen, w = generator(noise, return_w=True)
                disc_fake_gen_pred = discriminator(fake_imgs_gen)
                loss_gen_adv = F.binary_cross_entropy_with_logits(disc_fake_gen_pred, torch.ones_like(disc_fake_gen_pred))

                # Generate NCA images and get discriminator feedback
                seed = nca_model.get_seed(batch_size=real_imgs.shape[0], size=IMG_SIZE, device=DEVICE, seed_type="distributed")
                nca_steps = random.randint(NCA_STEPS_MIN, NCA_STEPS_MAX)
                nca_output_grid = nca_model(seed, w, steps=nca_steps, target_img=real_imgs)  # Pass target images for rich conditioning
                nca_rgba = nca_model.to_rgba(nca_output_grid)
                fake_imgs_nca = nca_rgba[:, :3, :, :]  # Already in [-1, 1] range from tanh
                disc_fake_nca_pred = discriminator(fake_imgs_nca)
                loss_nca_adv = F.binary_cross_entropy_with_logits(disc_fake_nca_pred, torch.ones_like(disc_fake_nca_pred))
                
                # Enhanced mutual evaluation using weighted signals
                gen_scores_nca = nca_evaluator(fake_imgs_gen)  # NCA evaluates Generator
                nca_scores_gen = gen_evaluator(fake_imgs_nca)  # Generator evaluates NCA
                
                # Incorporate cross-learning signals into evaluation
                gen_scores_enhanced = 0.6 * gen_scores_nca + 0.4 * weighted_gen_quality.detach()
                nca_scores_enhanced = 0.6 * nca_scores_gen + 0.4 * weighted_nca_quality.detach()
                
                # ADAPTIVE PENALTY MECHANISMS (replacing constant penalties)
                # Based on gradient optimization research for better convergence
                
                # 1. Curriculum-based penalty scaling (starts gentle, becomes stricter)
                curriculum_progress = min(epoch / 100.0, 1.0)  # 0 to 1 over 100 epochs
                base_penalty = 0.5 + curriculum_progress * 2.0  # 0.5 -> 2.5 scaling
                
                # 2. Performance-adaptive penalties (harder penalties for worse performance) - using enhanced scores
                gen_performance_factor = torch.clamp(1.0 - gen_scores_enhanced.mean(), 0.1, 2.0)
                nca_performance_factor = torch.clamp(1.0 - nca_scores_enhanced.mean(), 0.1, 2.0)
                
                # 3. Gradient-aware penalty scaling (prevent exploding/vanishing gradients)
                # Use parameter magnitudes as proxy for gradient health (since gradients not computed yet)
                gen_param_norm = torch.norm(torch.cat([p.flatten() for p in generator.parameters()]))
                nca_param_norm = torch.norm(torch.cat([p.flatten() for p in nca_model.parameters()]))
                
                # Adaptive scaling based on parameter health (prevent explosion/vanishing)
                gen_grad_scale = torch.clamp(10.0 / (gen_param_norm + 1e-8), 0.1, 5.0)
                nca_grad_scale = torch.clamp(10.0 / (nca_param_norm + 1e-8), 0.1, 5.0)
                
                # 4. Smooth penalty transitions (avoid sudden jumps)
                quality_threshold = 0.3 + curriculum_progress * 0.2  # Dynamic threshold: 0.3->0.5
                
                # Smooth penalty functions instead of hard thresholds - using enhanced scores
                gen_penalty_smooth = torch.sigmoid((quality_threshold - gen_scores_enhanced) * 10) * base_penalty * gen_performance_factor * gen_grad_scale
                nca_penalty_smooth = torch.sigmoid((quality_threshold - nca_scores_enhanced) * 10) * base_penalty * nca_performance_factor * nca_grad_scale
                
                # 5. Anti-saturation mechanism (prevent getting stuck in bad local minima) - using enhanced scores
                gen_improvement = torch.relu(gen_scores_enhanced.mean() - 0.1)  # Reward improvement
                nca_improvement = torch.relu(nca_scores_enhanced.mean() - 0.1)
                saturation_bonus_gen = torch.exp(-gen_improvement * 5) * 0.5  # Extra push when stuck
                saturation_bonus_nca = torch.exp(-nca_improvement * 5) * 0.5
                
                # 6. Feature matching with adaptive weight
                # Extract discriminator features for feature matching loss
                with torch.no_grad():
                    # Use discriminator's intermediate features for feature matching
                    disc_features_real = discriminator.model[:-1](real_imgs)  # All layers except final
                    disc_features_nca = discriminator.model[:-1](fake_imgs_nca)
                    disc_features_gen = discriminator.model[:-1](fake_imgs_gen)
                
                feature_weight = 0.5 + curriculum_progress * 1.0  # 0.5 -> 1.5 over training
                feature_match_loss_nca = F.mse_loss(disc_features_nca.mean(dim=0), disc_features_real.mean(dim=0)) * feature_weight
                feature_match_loss_gen = F.mse_loss(disc_features_gen.mean(dim=0), disc_features_real.mean(dim=0)) * feature_weight
                
                # 7. Confidence regulation (prevent overconfidence, allow healthy confidence) - using enhanced scores
                confidence_target = 0.7 + curriculum_progress * 0.2  # Target confidence grows with training
                gen_confidence_penalty = F.mse_loss(gen_scores_enhanced.mean(), torch.tensor(confidence_target, device=DEVICE)) * 0.5
                nca_confidence_penalty = F.mse_loss(nca_scores_enhanced.mean(), torch.tensor(confidence_target, device=DEVICE)) * 0.5
                
                # Combined adaptive losses (much more stable than constant penalties)
                loss_gen_mutual = (gen_penalty_smooth.mean() + 
                                 saturation_bonus_gen + 
                                 feature_match_loss_gen +
                                 gen_confidence_penalty)
                
                loss_nca_mutual = (nca_penalty_smooth.mean() + 
                                 saturation_bonus_nca + 
                                 feature_match_loss_nca +
                                 nca_confidence_penalty)

                # Combined losses with stronger penalties (removed matching loss)
                loss_gen_total = loss_gen_adv + 0.3 * loss_gen_mutual  # Increased weight
                loss_nca_total = loss_nca_adv + 0.3 * loss_nca_mutual  # Removed matching loss
                
                loss_total = loss_gen_total + loss_nca_total
                loss_total.backward()
                
                opt_gen.step()
                opt_nca.step()
                
                # Historical averaging update (damping force for convergence)
                with torch.no_grad():
                    for name, model in [('gen', generator), ('nca', nca_model)]:
                        if historical_params[name] is None:
                            # Initialize historical parameters
                            historical_params[name] = {key: param.clone() for key, param in model.named_parameters()}
                        else:
                            # Update historical average (for monitoring purposes only)
                            for key, param in model.named_parameters():
                                historical_params[name][key] = (historical_decay * historical_params[name][key] + 
                                                               (1 - historical_decay) * param.detach())

                # Accumulate scores for this epoch - using enhanced cross-learning scores
                with torch.no_grad():
                    epoch_scores['gen_quality'] += gen_scores_enhanced.mean().item()
                    epoch_scores['nca_quality'] += nca_scores_enhanced.mean().item()
                    epoch_scores['gen_eval_loss'] += loss_gen_eval.item()
                    epoch_scores['nca_eval_loss'] += loss_nca_eval.item()
                    epoch_scores['transformer_loss'] += loss_transformer.item()
                    epoch_scores['cross_learning_loss'] = epoch_scores.get('cross_learning_loss', 0) + cross_learning_loss.item()
                    epoch_scores['ensemble_prediction'] = epoch_scores.get('ensemble_prediction', 0) + ensemble_prediction.mean().item()
                    
                    # Track signal weights for analysis
                    gen_weights = cross_learning_results['gen_weights'].mean(dim=0)
                    nca_weights = cross_learning_results['nca_weights'].mean(dim=0)
                    epoch_scores['gen_signal_weights'] = gen_weights.cpu().numpy().tolist()
                    epoch_scores['nca_signal_weights'] = nca_weights.cpu().numpy().tolist()
                    
                    # Additional penalty tracking for monitoring
                    epoch_scores['gen_penalty'] = epoch_scores.get('gen_penalty', 0) + loss_gen_mutual.item()
                    epoch_scores['nca_penalty'] = epoch_scores.get('nca_penalty', 0) + loss_nca_mutual.item()
                    epoch_scores['feature_match_gen'] = epoch_scores.get('feature_match_gen', 0) + feature_match_loss_gen.item()
                    epoch_scores['feature_match_nca'] = epoch_scores.get('feature_match_nca', 0) + feature_match_loss_nca.item()
                    
                    # Transformer-specific scores
                    if transformer_critic.mode == "critic":
                        epoch_scores['transformer_real_score'] = epoch_scores.get('transformer_real_score', 0) + real_transformer_score.mean().item()
                        epoch_scores['transformer_gen_score'] = epoch_scores.get('transformer_gen_score', 0) + gen_transformer_score.mean().item()
                        epoch_scores['transformer_nca_score'] = epoch_scores.get('transformer_nca_score', 0) + nca_transformer_score.mean().item()

                # Update status for UI
                if batch_idx % 10 == 0:
                    current_scores = {
                        'gen_quality': gen_scores_enhanced.mean().item(),
                        'nca_quality': nca_scores_enhanced.mean().item(),
                        'ensemble_quality': ensemble_prediction.mean().item(),
                        'cross_learning_loss': cross_learning_loss.item()
                    }
                    
                    # Add current metrics to history for mini-graphs
                    current_metrics = {
                        'loss_disc': loss_disc.item(),
                        'loss_gen': loss_gen_adv.item(),
                        'loss_nca': loss_nca_adv.item(),
                        'gen_quality': current_scores['gen_quality'],
                        'nca_quality': current_scores['nca_quality'],
                        'ensemble_quality': current_scores['ensemble_quality']
                    }
                    metrics_history.append(current_metrics)
                    
                    # Generate mini-graphs for key metrics
                    mini_graphs = format_mini_graphs(
                        metrics_history, 
                        ['loss_disc', 'loss_gen', 'loss_nca', 'gen_quality', 'nca_quality'],
                        width=6
                    )
                    
                    status_text = (
                        f"Epoch {epoch+1}/{EPOCHS}, "
                        f"Loss D: {loss_disc.item():.3f}, "
                        f"Loss G: {loss_gen_adv.item():.3f}, "
                        f"Loss NCA: {loss_nca_adv.item():.3f}, "
                        f"Transformer ({transformer_critic.mode}): {loss_transformer.item():.3f}, "
                        f"Cross-Learning: {current_scores['cross_learning_loss']:.3f}, "
                        f"Gen Quality: {current_scores['gen_quality']:.3f}, "
                        f"NCA Quality: {current_scores['nca_quality']:.3f}, "
                        f"Ensemble: {current_scores['ensemble_quality']:.3f}"
                    )
                    
                    # Add mini-graphs underneath the main status
                    if mini_graphs:
                        status_text += f"\n📊 {mini_graphs}"
                    
                    progress_bar.set_postfix_str(status_text)
                    
                    with torch.no_grad():
                        sample_gen, sample_w = generator(fixed_noise, return_w=True)
                        seed = nca_model.get_seed(batch_size=fixed_noise.shape[0], size=IMG_SIZE, device=DEVICE, seed_type="distributed")
                        sample_nca_grid = nca_model(seed, sample_w, steps=NCA_STEPS_MAX, target_img=real_imgs)  # Pass target for visualization
                        sample_nca_rgba = nca_model.to_rgba(sample_nca_grid)
                        sample_nca = sample_nca_rgba[:, :3, :, :]  # Already in [-1, 1] range
                        
                        # Debug NCA output
                        print(f"NCA grid stats: min={sample_nca_grid.min():.3f}, max={sample_nca_grid.max():.3f}")
                        print(f"NCA RGBA stats: min={sample_nca_rgba.min():.3f}, max={sample_nca_rgba.max():.3f}")
                        print(f"Final NCA stats: min={sample_nca.min():.3f}, max={sample_nca.max():.3f}")
                        
                        # Check alpha channel specifically
                        alpha_stats = sample_nca_rgba[:, 3:4, :, :]
                        alive_cells = (alpha_stats > 0.1).sum().item()
                        total_cells = alpha_stats.numel()
                        print(f"Alpha channel: alive_cells={alive_cells}/{total_cells} ({100*alive_cells/total_cells:.1f}%)")
                        
                        if torch.isnan(sample_nca).any() or torch.isinf(sample_nca).any():
                            print(f"WARNING: NCA output contains NaN or Inf values!")
                        
                        if alive_cells == 0:
                            print(f"WARNING: NCA has completely died! No alive cells remaining.")

                    # Pass images as a list of tensors: [target, generator, nca]
                    update_status(status_text, images=[real_imgs, sample_gen, sample_nca], scores=current_scores)

            # Average scores for this epoch (skip string values and lists)
            for key in epoch_scores:
                if isinstance(epoch_scores[key], (int, float)) and not isinstance(epoch_scores[key], str):
                    epoch_scores[key] /= batch_count
            
            score_history.append(epoch_scores)
            
            # Save checkpoint more frequently in early epochs, then every 5 epochs
            save_checkpoint_now = False
            if epoch < 5:  # First 5 epochs, save after each epoch
                save_checkpoint_now = True
            elif (epoch + 1) % 5 == 0:  # Then every 5 epochs
                save_checkpoint_now = True
                
            if save_checkpoint_now:
                save_checkpoint(epoch + 1, models, optimizers, loss_history, score_history, CHECKPOINT_DIR, KEEP_CHECKPOINTS)

        update_status("Training finished.")
        save_checkpoint(EPOCHS, models, optimizers, loss_history, score_history, CHECKPOINT_DIR, KEEP_CHECKPOINTS)
        
    except Exception as e:
        import traceback
        error_msg = f"Error in training loop: {str(e)}"
        print(error_msg)
        print("Full traceback:")
        traceback.print_exc()
        update_status(error_msg, error=True)

# --- Transformer Critic/Imitator ---
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
            nn.Linear(embed_dim * 2, self.num_patches * 3),  # RGB for each patch
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
        """Switch between critic and imitator modes"""
        assert mode in ["critic", "imitator"], "Mode must be 'critic' or 'imitator'"
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
            
        elif self.mode == "imitator":
            # Use all patch tokens for generation
            if target is None:
                raise ValueError("Target image required for imitator mode")
            
            # Extract patch features (excluding CLS token)
            patch_features = x[:, 1:]  # [B, num_patches, embed_dim]
            
            # Generate imitation
            generated_patches = self.imitator_head(patch_features)  # [B, num_patches, 3]
            
            # Reshape to image format
            patches_per_side = int(self.num_patches ** 0.5)
            generated_patches = generated_patches.view(B, patches_per_side, patches_per_side, 3)
            generated_patches = generated_patches.permute(0, 3, 1, 2)  # [B, 3, H//patch_size, W//patch_size]
            
            # Upsample to original image size
            generated_img = F.interpolate(generated_patches, size=(H, W), mode='bilinear', align_corners=False)
            
            # Compute imitation loss
            imitation_loss = F.mse_loss(generated_img, target)
            
            return generated_img, imitation_loss

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

# --- Enhanced Cross-Learning with Signal Weighting ---
class SignalWeightingNetwork(nn.Module):
    """
    Adaptive signal weighting system that learns optimal weights for combining
    multiple evaluation signals from different models.
    """
    def __init__(self, num_signals=5, hidden_dim=64):
        super().__init__()
        self.num_signals = num_signals
        
        # Context encoder: learns to understand the current training state
        self.context_encoder = nn.Sequential(
            nn.Linear(num_signals + 3, hidden_dim),  # +3 for epoch, batch, time context
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_signals),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )
        
        # Signal confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_signals, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_signals),
            nn.Sigmoid()  # Confidence scores [0,1]
        )
        
        # Performance tracker for adaptive weighting
        self.register_buffer('signal_performance_history', torch.zeros(num_signals, 100))  # Last 100 batches
        self.register_buffer('history_index', torch.tensor(0))
        
    def update_performance_history(self, signal_scores, actual_performance):
        """Update performance history for each signal"""
        with torch.no_grad():
            idx = self.history_index.item() % 100
            # Calculate how well each signal predicted actual performance
            # signal_scores: [num_signals, batch_size], actual_performance: [batch_size]
            signal_accuracy = 1.0 - torch.abs(signal_scores - actual_performance.unsqueeze(0))
            # Take mean across batch dimension and flatten to [num_signals]
            accuracy_per_signal = signal_accuracy.mean(dim=1).squeeze()
            if accuracy_per_signal.dim() == 0:  # Handle single signal case
                accuracy_per_signal = accuracy_per_signal.unsqueeze(0)
            self.signal_performance_history[:, idx] = accuracy_per_signal
            self.history_index += 1
    
    def forward(self, signals, epoch_progress, batch_progress):
        """
        Args:
            signals: [num_signals, batch_size] - scores from different evaluators
            epoch_progress: float - progress through current epoch [0,1]
            batch_progress: float - progress through training [0,1]
        """
        batch_size = signals.shape[1]
        
        # Create context vector - ensure it's [batch_size, 3]
        context = torch.tensor([
            epoch_progress,
            batch_progress,
            signals.std().item()  # Signal diversity as context
        ], device=signals.device)
        # Reshape to [1, 3] then repeat to [batch_size, 3]
        context = context.view(1, -1).repeat(batch_size, 1)
        
        # Combine signals with context for weight prediction
        # signals.mean(dim=1) gives [num_signals], we want [batch_size, num_signals]
        signal_means = signals.mean(dim=1)  # [num_signals]
        signal_means = signal_means.view(1, -1).repeat(batch_size, 1)  # [batch_size, num_signals]
        weight_input = torch.cat([signal_means, context], dim=1)
        
        # Get adaptive weights
        adaptive_weights = self.context_encoder(weight_input)  # [batch_size, num_signals]
        
        # Get confidence scores
        confidence_scores = self.confidence_estimator(signal_means)  # [batch_size, num_signals]
        
        # Historical performance weighting
        if self.history_index > 10:  # Only after some history
            historical_performance = self.signal_performance_history.mean(dim=1)  # [num_signals]
            historical_weights = F.softmax(historical_performance * 2.0, dim=0)  # Sharpen distribution
            # historical_weights is [num_signals], we want [batch_size, num_signals]
            historical_weights = historical_weights.view(1, -1).repeat(batch_size, 1)
        else:
            historical_weights = torch.ones_like(adaptive_weights) / self.num_signals
        
        # Combine adaptive weights with historical performance and confidence
        final_weights = (
            0.4 * adaptive_weights +           # 40% current context
            0.3 * historical_weights +         # 30% historical performance  
            0.3 * confidence_scores            # 30% confidence
        )
        
        # Normalize to sum to 1
        final_weights = F.softmax(final_weights * 2.0, dim=1)  # Sharpen
        
        # Apply weights to signals
        weighted_signals = (signals.mT * final_weights).sum(dim=1)  # [batch_size]
        
        return weighted_signals, final_weights, confidence_scores

class EnhancedCrossLearningSystem(nn.Module):
    """
    Enhanced system that enables all models to learn from each other's outputs
    with sophisticated signal weighting and cross-model knowledge transfer.
    """
    def __init__(self, img_size=64, num_models=5):
        super().__init__()
        self.num_models = num_models
        
        # Signal weighting networks for different learning objectives
        self.quality_weighter = SignalWeightingNetwork(num_signals=num_models)
        self.style_weighter = SignalWeightingNetwork(num_signals=num_models)
        self.content_weighter = SignalWeightingNetwork(num_signals=num_models)
        
        # Cross-model feature extractors
        self.feature_extractors = nn.ModuleDict({
            'discriminator': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 64, 128)
            ),
            'generator': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 64, 128)
            ),
            'nca': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 64, 128)
            ),
            'transformer': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 64, 128)
            )
        })
        
        # Cross-model learning networks
        self.cross_learners = nn.ModuleDict({
            'gen_from_nca': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'nca_from_gen': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'both_from_transformer': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # Outputs for both gen and nca
                nn.Sigmoid()
            ),
            'all_from_discriminator': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # Outputs for gen, nca, transformer
                nn.Sigmoid()
            )
        })
        
        # Ensemble predictor that combines all signals
        self.ensemble_predictor = nn.Sequential(
            nn.Linear(num_models, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def extract_cross_features(self, images_dict):
        """Extract features from images using cross-model feature extractors"""
        features = {}
        for model_name, images in images_dict.items():
            if model_name in self.feature_extractors:
                features[model_name] = self.feature_extractors[model_name](images)
        return features
    
    def compute_cross_learning_signals(self, features):
        """Compute cross-learning signals between models"""
        signals = {}
        
        # Generator learns from NCA
        if 'nca' in features:
            signals['gen_from_nca'] = self.cross_learners['gen_from_nca'](features['nca'])
        
        # NCA learns from Generator  
        if 'generator' in features:
            signals['nca_from_gen'] = self.cross_learners['nca_from_gen'](features['generator'])
        
        # Both learn from Transformer
        if 'transformer' in features:
            transformer_signals = self.cross_learners['both_from_transformer'](features['transformer'])
            signals['gen_from_transformer'] = transformer_signals[:, 0:1]
            signals['nca_from_transformer'] = transformer_signals[:, 1:2]
        
        # All learn from Discriminator
        if 'discriminator' in features:
            disc_signals = self.cross_learners['all_from_discriminator'](features['discriminator'])
            signals['gen_from_discriminator'] = disc_signals[:, 0:1]
            signals['nca_from_discriminator'] = disc_signals[:, 1:2]
            signals['transformer_from_discriminator'] = disc_signals[:, 2:3]
        
        return signals
    
    def forward(self, images_dict, epoch_progress, batch_progress, target_performance=None):
        """
        Args:
            images_dict: {'real': tensor, 'generator': tensor, 'nca': tensor, 'transformer': tensor}
            epoch_progress: float [0,1]
            batch_progress: float [0,1]
            target_performance: ground truth performance for updating weights
        """
        # Extract cross-model features
        features = self.extract_cross_features(images_dict)
        
        # Compute cross-learning signals
        cross_signals = self.compute_cross_learning_signals(features)
        
        # Organize signals by target model
        gen_signals = torch.stack([
            cross_signals.get('gen_from_nca', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            cross_signals.get('gen_from_transformer', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            cross_signals.get('gen_from_discriminator', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            torch.ones_like(cross_signals[list(cross_signals.keys())[0]]) * 0.5,  # Baseline
            torch.ones_like(cross_signals[list(cross_signals.keys())[0]]) * 0.5   # Baseline
        ], dim=0)  # [5, batch_size]
        
        nca_signals = torch.stack([
            cross_signals.get('nca_from_gen', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            cross_signals.get('nca_from_transformer', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            cross_signals.get('nca_from_discriminator', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            torch.ones_like(cross_signals[list(cross_signals.keys())[0]]) * 0.5,  # Baseline
            torch.ones_like(cross_signals[list(cross_signals.keys())[0]]) * 0.5   # Baseline
        ], dim=0)  # [5, batch_size]
        
        # Apply signal weighting
        weighted_gen_quality, gen_weights, gen_confidence = self.quality_weighter(
            gen_signals, epoch_progress, batch_progress
        )
        weighted_nca_quality, nca_weights, nca_confidence = self.quality_weighter(
            nca_signals, epoch_progress, batch_progress
        )
        
        # Ensemble prediction - ensure all tensors have the same shape [batch_size]
        transformer_signal = cross_signals.get('transformer_from_discriminator', torch.zeros_like(weighted_gen_quality))
        # Ensure transformer_signal is [batch_size] shape
        if transformer_signal.dim() > 1:
            transformer_signal = transformer_signal.squeeze()
        if transformer_signal.dim() == 0:  # scalar
            transformer_signal = transformer_signal.expand_as(weighted_gen_quality)
        
        all_signals = torch.stack([
            weighted_gen_quality,
            weighted_nca_quality,
            transformer_signal,
            torch.ones_like(weighted_gen_quality) * 0.5,  # Baseline
            torch.ones_like(weighted_gen_quality) * 0.5   # Baseline
        ], dim=1)  # [batch_size, 5]
        
        ensemble_prediction = self.ensemble_predictor(all_signals)
        
        # Update performance history if target provided
        if target_performance is not None:
            self.quality_weighter.update_performance_history(gen_signals, target_performance)
            self.quality_weighter.update_performance_history(nca_signals, target_performance)
        
        return {
            'weighted_gen_quality': weighted_gen_quality,
            'weighted_nca_quality': weighted_nca_quality,
            'ensemble_prediction': ensemble_prediction.squeeze(),
            'gen_weights': gen_weights,
            'nca_weights': nca_weights,
            'gen_confidence': gen_confidence,
            'nca_confidence': nca_confidence,
            'cross_signals': cross_signals
        }

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