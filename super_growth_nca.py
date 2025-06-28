import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import time
import os
from datetime import datetime

# CPU Optimizations
torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision('high')

class SuperGrowthNCA(nn.Module):
    """Ultra-aggressive growth NCA designed to overcome death spiral"""
    def __init__(self, n_channels=8, hidden_size=64):
        super().__init__()
        self.n_channels = n_channels
        
        # Perception kernels
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Larger network
        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 3, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Small dropout for regularization
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, n_channels, 1),
            nn.Tanh()
        )
        
        # ULTRA-AGGRESSIVE growth parameters
        self.base_growth = nn.Parameter(torch.tensor(1.0))  # Much higher base
        self.survival_threshold = nn.Parameter(torch.tensor(0.05))  # Lower threshold
        self.resurrection_factor = nn.Parameter(torch.tensor(0.2))  # Allow resurrection
        
        # Initialize for maximum growth
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.5)  # Aggressive initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Positive bias
    
    def perceive(self, x):
        """Simple but effective perception"""
        sobel_x_expanded = self.sobel_x.repeat(self.n_channels, 1, 1, 1)
        sobel_y_expanded = self.sobel_y.repeat(self.n_channels, 1, 1, 1)
        
        dx = F.conv2d(x, sobel_x_expanded, groups=self.n_channels, padding=1)
        dy = F.conv2d(x, sobel_y_expanded, groups=self.n_channels, padding=1)
        
        return torch.cat([x, dx, dy], dim=1)
    
    def forward(self, x, force_growth=True):
        """Ultra-aggressive growth forward pass"""
        perceived = self.perceive(x)
        dx = self.update_net(perceived)
        
        # Current state analysis
        alive_mask = (x[:, 3:4] > self.survival_threshold).float()
        current_alive_ratio = alive_mask.mean()
        
        # ULTRA-AGGRESSIVE GROWTH STRATEGY
        if force_growth and current_alive_ratio < 0.3:
            # Emergency growth mode - prevent death at all costs
            growth_multiplier = 5.0  # Extreme multiplier
            
            # Encourage growth everywhere
            neighbor_life = F.avg_pool2d(alive_mask, 3, stride=1, padding=1)
            growth_zones = (neighbor_life > 0.01).float()  # Very permissive
            
            # Massive growth boost
            dx = dx * growth_multiplier
            
            # Direct alpha channel boosting in growth zones - avoid in-place ops
            alpha_boost = growth_zones * 0.3  # Direct injection
            dx_alpha = dx[:, 3:4] + alpha_boost
            dx = torch.cat([dx[:, :3], dx_alpha, dx[:, 4:]], dim=1)
            
            # Resurrection mechanism - allow dead cells to come alive
            resurrection_mask = (neighbor_life > 0.1).float() * (1 - alive_mask)
            resurrection_boost = resurrection_mask * self.resurrection_factor
            dx_alpha_final = dx[:, 3:4] + resurrection_boost
            dx = torch.cat([dx[:, :3], dx_alpha_final, dx[:, 4:]], dim=1)
        
        # Apply base growth
        new_x = x + dx * self.base_growth
        
        # Minimal death constraints - only prevent extreme values
        new_x = torch.clamp(new_x, -2, 2)  # Wider bounds
        
        # Ensure alpha channel doesn't go negative (but allow high values)
        new_x[:, 3:4] = torch.clamp(new_x[:, 3:4], 0, 2)
        
        return new_x

def create_large_target(size=32):
    """Create a large target to encourage maximum growth"""
    target = torch.zeros(1, 8, size, size)
    
    # Nearly fill the entire space
    center = size // 2
    radius = size // 2.5  # Very large circle
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    mask = ((x - center) ** 2 + (y - center) ** 2) < radius ** 2
    
    target[0, 0, mask] = 1.0  # Red
    target[0, 3, mask] = 1.0  # Alpha
    
    return target

def initialize_super_seed(size=32):
    """Initialize with maximum viable seeds"""
    state = torch.zeros(1, 8, size, size)
    center = size // 2
    
    # Large cross-shaped seed for maximum growth potential
    thickness = 3
    
    # Horizontal bar
    state[0, 3, center-thickness:center+thickness, center-8:center+9] = 1.0
    state[0, 0, center-1:center+2, center-4:center+5] = 0.8
    
    # Vertical bar  
    state[0, 3, center-8:center+9, center-thickness:center+thickness] = 1.0
    state[0, 1, center-4:center+5, center-1:center+2] = 0.8
    
    # Central intersection
    state[0, 3, center-1:center+2, center-1:center+2] = 1.0
    state[0, 2, center, center] = 1.0  # Blue center
    
    return state

def train_super_growth():
    """Super-aggressive growth training"""
    print("💥 SUPER-AGGRESSIVE Growth NCA Training")
    print("=" * 50)
    print("🎯 Goal: Force NCA to grow by any means necessary!")
    print("🚀 Strategy: Maximum growth incentives, minimal death constraints")
    print()
    
    device = torch.device('cpu')
    size = 32
    
    # Initialize
    model = SuperGrowthNCA(8, 64).to(device)
    target = create_large_target(size)
    optimizer = optim.Adam(model.parameters(), lr=0.005)  # Higher learning rate
    
    target_alive_ratio = (target[:, 3:4] > 0.1).float().mean().item()
    print(f"🎯 Target alive ratio: {target_alive_ratio:.3f} (very ambitious!)")
    
    # Single aggressive stage - focus on growth only
    epochs = 100
    results = []
    
    print(f"\n💥 ULTRA-AGGRESSIVE GROWTH PHASE")
    print(f"   Strategy: Force growth, ignore stability initially")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Initialize with super seed
        state = initialize_super_seed(size)
        
        # Run with maximum growth
        for step in range(20):  # Shorter runs to prevent explosion
            state = model(state, force_growth=True)
        
        # Calculate loss - focus mainly on alive ratio
        alive_mask = (state[:, 3:4] > 0.1).float()
        current_alive = alive_mask.mean()
        
        # Growth-focused loss
        growth_loss = F.mse_loss(current_alive, torch.tensor(0.3))  # Target 30% alive
        pattern_loss = F.mse_loss(state[:, :4], target[:, :4])
        
        # Heavily weight growth over pattern
        total_loss = 0.8 * growth_loss + 0.2 * pattern_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # High gradient clip
        optimizer.step()
        
        results.append({
            "epoch": epoch,
            "alive_ratio": current_alive.item(),
            "growth_loss": growth_loss.item(),
            "pattern_loss": pattern_loss.item(),
            "total_loss": total_loss.item()
        })
        
        # Frequent progress reports
        if epoch % 5 == 0:
            print(f"   Epoch {epoch:3d}: Alive={current_alive.item():.4f}, "
                  f"Growth Loss={growth_loss.item():.4f}, "
                  f"Pattern={pattern_loss.item():.4f}")
        
        # Early success detection
        if current_alive.item() > 0.2:
            print(f"   🎉 BREAKTHROUGH! Achieved {current_alive.item():.3f} alive ratio!")
            if current_alive.item() > 0.25:
                print(f"   🚀 EXCELLENT! Strong growth achieved early!")
        
        # Emergency stop if complete death
        if current_alive.item() < 0.001 and epoch > 20:
            print(f"   ⚠️  Death detected at epoch {epoch}, but continuing...")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n🎯 Final Evaluation")
    model.eval()
    
    with torch.no_grad():
        # Test final state
        final_state = initialize_super_seed(size)
        final_states = [final_state.clone()]
        
        for step in range(50):
            final_state = model(final_state, force_growth=False)  # Test without force
            if step % 10 == 0:
                final_states.append(final_state.clone())
        
        final_alive = (final_state[:, 3:4] > 0.1).float().mean().item()
        final_pattern_loss = F.mse_loss(final_state[:, :4], target[:, :4]).item()
    
    # Results analysis
    alive_ratios = [r["alive_ratio"] for r in results]
    max_alive = max(alive_ratios)
    final_alive_training = alive_ratios[-1]
    
    print(f"   📊 Training Results:")
    print(f"      Max alive ratio: {max_alive:.4f}")
    print(f"      Final training alive: {final_alive_training:.4f}")
    print(f"      Final test alive: {final_alive:.4f}")
    print(f"      Pattern loss: {final_pattern_loss:.4f}")
    print(f"      Training time: {training_time:.1f}s")
    
    # Success assessment
    growth_breakthrough = max_alive > 0.15
    sustained_growth = final_alive > 0.1
    overall_success = growth_breakthrough and sustained_growth
    
    print(f"\n🏆 SUCCESS ASSESSMENT:")
    print(f"   🌱 Growth Breakthrough: {'✅' if growth_breakthrough else '❌'} (max: {max_alive:.3f})")
    print(f"   🔄 Sustained Growth: {'✅' if sustained_growth else '❌'} (final: {final_alive:.3f})")
    print(f"   🎯 Overall Success: {'✅' if overall_success else '❌'}")
    
    if overall_success:
        print(f"\n🎉 VICTORY! Super-aggressive approach achieved stable NCA growth!")
        print(f"   The NCA learned to grow and maintain living populations!")
    elif growth_breakthrough:
        print(f"\n🔄 PARTIAL SUCCESS! Achieved growth but couldn't sustain it")
        print(f"   Need to work on stability while maintaining growth")
    else:
        print(f"\n❌ CHALLENGE REMAINS: Even ultra-aggressive approach struggled")
        print(f"   May need fundamental architecture changes")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results = {
        "approach": "Super-Aggressive Growth",
        "training_results": results,
        "summary": {
            "max_alive_ratio": max_alive,
            "final_training_alive": final_alive_training,
            "final_test_alive": final_alive,
            "final_pattern_loss": final_pattern_loss,
            "training_time": training_time,
            "growth_breakthrough": bool(growth_breakthrough),
            "sustained_growth": bool(sustained_growth),
            "overall_success": bool(overall_success)
        }
    }
    
    with open(f'super_growth_results_{timestamp}.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save states
    np.save(f'super_growth_states_{timestamp}.npy', torch.stack(final_states).numpy())
    
    print(f"\n💾 Results saved to super_growth_results_{timestamp}.json")
    
    return final_results, final_states

if __name__ == "__main__":
    print("🧬 SUPER-AGGRESSIVE NCA Growth Experiment")
    print("   Hypothesis: Maximum growth incentives can overcome death spiral")
    print("   Approach: Force growth by any means necessary")
    print()
    
    results, states = train_super_growth() 