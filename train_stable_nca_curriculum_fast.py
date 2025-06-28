import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from datetime import datetime

# CPU Optimizations
torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision('high')

class OptimizedStableNCA(nn.Module):
    """Optimized version with reduced complexity for faster training"""
    def __init__(self, n_channels=8, hidden_size=32):  # Reduced from 16 channels, 64 hidden
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        
        # Simplified perception - just Sobel + identity
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Much smaller network
        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 3, hidden_size, 1),  # 3 = original + 2 gradients
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, n_channels, 1),
            nn.Tanh()
        )
        
        # Learnable stability parameters
        self.growth_rate = nn.Parameter(torch.tensor(0.1))
        self.death_rate = nn.Parameter(torch.tensor(0.05))
        self.stability_factor = nn.Parameter(torch.tensor(0.8))
        
        # Initialize for stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)  # Small initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def perceive(self, x):
        """Fast perception with minimal operations"""
        # Apply gradients to all channels at once
        dx = F.conv2d(x, self.sobel_x.expand(self.n_channels, 1, 1, 1), groups=self.n_channels, padding=1)
        dy = F.conv2d(x, self.sobel_y.expand(self.n_channels, 1, 1, 1), groups=self.n_channels, padding=1)
        return torch.cat([x, dx, dy], dim=1)
    
    def forward(self, x, target_alive_ratio=0.1):
        """Optimized forward pass with stability control"""
        # Fast perception
        perceived = self.perceive(x)
        
        # Update
        dx = self.update_net(perceived)
        
        # Current alive ratio
        alive_mask = (x[:, 3:4] > 0.1).float()
        current_alive_ratio = alive_mask.mean()
        
        # Dynamic stability control
        if current_alive_ratio < target_alive_ratio * 0.5:
            # Too few alive - encourage growth
            growth_boost = 1.5
            death_penalty = 0.5
        elif current_alive_ratio > target_alive_ratio * 2.0:
            # Too many alive - encourage death
            growth_boost = 0.5
            death_penalty = 1.5
        else:
            # In target range - maintain stability
            growth_boost = 1.0
            death_penalty = 1.0
        
        # Apply scaled update
        scale = self.stability_factor * growth_boost
        dx = dx * scale
        
        # Apply death penalty to overpopulated areas
        if death_penalty > 1.0:
            density = F.avg_pool2d(alive_mask, 3, stride=1, padding=1)
            death_mask = (density > 0.7).float()  # High density areas
            dx = dx * (1 - death_mask * (death_penalty - 1) * 0.3)
        
        new_x = x + dx
        
        # Enforce alive mask and bounds
        new_x = new_x * alive_mask
        new_x = torch.clamp(new_x, -1, 1)
        
        return new_x

def create_target_pattern(size=32, pattern_type="circle"):
    """Create simple target patterns for fast training"""
    target = torch.zeros(1, 8, size, size)
    
    if pattern_type == "circle":
        center = size // 2
        radius = size // 4
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        mask = ((x - center) ** 2 + (y - center) ** 2) < radius ** 2
        target[0, 0, mask] = 1.0  # Red
        target[0, 3, mask] = 1.0  # Alpha
    elif pattern_type == "cross":
        center = size // 2
        thickness = 2
        target[0, 1, center-thickness:center+thickness, :] = 1.0  # Green horizontal
        target[0, 1, :, center-thickness:center+thickness] = 1.0  # Green vertical
        target[0, 3, center-thickness:center+thickness, :] = 1.0  # Alpha horizontal
        target[0, 3, :, center-thickness:center+thickness] = 1.0  # Alpha vertical
    
    return target

def initialize_seed(size=32, seed_type="center"):
    """Initialize with small seed"""
    state = torch.zeros(1, 8, size, size)
    center = size // 2
    
    if seed_type == "center":
        state[0, 3, center-1:center+2, center-1:center+2] = 1.0  # Alpha
        state[0, 0, center, center] = 1.0  # Red center
    
    return state

def curriculum_stages():
    """Define 5 progressive curriculum stages"""
    return [
        {"name": "Stability", "target_alive": 0.05, "epochs": 50, "loss_weight": {"stability": 1.0, "pattern": 0.1}},
        {"name": "Growth", "target_alive": 0.1, "epochs": 50, "loss_weight": {"stability": 0.8, "pattern": 0.3}},
        {"name": "Control", "target_alive": 0.15, "epochs": 50, "loss_weight": {"stability": 0.6, "pattern": 0.5}},
        {"name": "Pattern", "target_alive": 0.2, "epochs": 50, "loss_weight": {"stability": 0.4, "pattern": 0.8}},
        {"name": "Mastery", "target_alive": 0.25, "epochs": 50, "loss_weight": {"stability": 0.2, "pattern": 1.0}}
    ]

def train_fast_curriculum():
    """Fast curriculum training with optimizations"""
    device = torch.device('cpu')
    size = 32  # Smaller size for speed
    
    print("🚀 Fast Stable NCA Curriculum Training")
    print("=" * 50)
    
    # Initialize
    model = OptimizedStableNCA(n_channels=8, hidden_size=32).to(device)
    target = create_target_pattern(size, "circle")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    stages = curriculum_stages()
    results = {"stages": [], "metrics": []}
    
    total_start_time = time.time()
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n📚 Stage {stage_idx + 1}: {stage['name']}")
        print(f"Target alive ratio: {stage['target_alive']}")
        
        stage_start_time = time.time()
        stage_losses = []
        stage_alive_ratios = []
        
        for epoch in range(stage['epochs']):
            model.train()
            optimizer.zero_grad()
            
            # Initialize seed
            state = initialize_seed(size)
            
            # Run NCA for fewer steps (speed optimization)
            n_steps = 20  # Reduced from 64
            for step in range(n_steps):
                state = model(state, target_alive_ratio=stage['target_alive'])
            
            # Calculate losses
            pattern_loss = F.mse_loss(state[:, :4], target[:, :4])  # Only RGBA channels
            
            alive_mask = (state[:, 3:4] > 0.1).float()
            current_alive_ratio = alive_mask.mean()
            
            # Stability loss - penalize deviation from target
            stability_loss = F.mse_loss(current_alive_ratio, torch.tensor(stage['target_alive']))
            
            # Combined loss
            total_loss = (stage['loss_weight']['pattern'] * pattern_loss + 
                         stage['loss_weight']['stability'] * stability_loss)
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            stage_losses.append(total_loss.item())
            stage_alive_ratios.append(current_alive_ratio.item())
            
            # Fast progress reporting
            if epoch % 10 == 0:
                print(f"Epoch {epoch:2d}: Loss={total_loss.item():.4f}, "
                      f"Alive={current_alive_ratio.item():.3f}, "
                      f"Target={stage['target_alive']:.3f}")
        
        stage_time = time.time() - stage_start_time
        avg_loss = np.mean(stage_losses[-10:])  # Last 10 epochs
        avg_alive = np.mean(stage_alive_ratios[-10:])
        
        stage_result = {
            "stage": stage['name'],
            "avg_loss": avg_loss,
            "avg_alive_ratio": avg_alive,
            "target_alive": stage['target_alive'],
            "time_seconds": stage_time,
            "converged": abs(avg_alive - stage['target_alive']) < 0.02
        }
        
        results['stages'].append(stage_result)
        
        print(f"✅ Stage {stage['name']} completed in {stage_time:.1f}s")
        print(f"   Final: Loss={avg_loss:.4f}, Alive={avg_alive:.3f}, "
              f"Converged={'Yes' if stage_result['converged'] else 'No'}")
    
    total_time = time.time() - total_start_time
    
    # Final evaluation
    print(f"\n🎯 Final Evaluation")
    model.eval()
    with torch.no_grad():
        final_state = initialize_seed(size)
        states = [final_state.clone()]
        
        for step in range(40):  # Longer evaluation
            final_state = model(final_state, target_alive_ratio=0.2)
            if step % 5 == 0:
                states.append(final_state.clone())
    
    final_alive = (final_state[:, 3:4] > 0.1).float().mean().item()
    final_loss = F.mse_loss(final_state[:, :4], target[:, :4]).item()
    
    print(f"Final alive ratio: {final_alive:.3f}")
    print(f"Final pattern loss: {final_loss:.4f}")
    print(f"Total training time: {total_time:.1f}s")
    
    # Save results
    results['summary'] = {
        "total_time": total_time,
        "final_alive_ratio": final_alive,
        "final_pattern_loss": final_loss,
        "all_stages_converged": all(s['converged'] for s in results['stages'])
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'fast_curriculum_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save final states for visualization
    np.save(f'fast_curriculum_states_{timestamp}.npy', 
            torch.stack(states).numpy())
    
    print(f"\n📊 Results saved to fast_curriculum_results_{timestamp}.json")
    return results, states

if __name__ == "__main__":
    results, states = train_fast_curriculum() 