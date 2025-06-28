import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import time
from datetime import datetime

# CPU Optimizations
torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')

class SimpleGrowthNCA(nn.Module):
    """Clean, simple NCA focused purely on growth"""
    def __init__(self, n_channels=8):
        super().__init__()
        self.n_channels = n_channels
        
        # Simple perception
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Simple update network
        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 3, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, n_channels, 1)
        )
        
        # Growth parameters
        self.growth_rate = nn.Parameter(torch.tensor(0.3))
        self.alive_threshold = 0.1
        
        # Initialize for growth
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.05)  # Slight positive bias
    
    def perceive(self, x):
        """Simple perception"""
        sobel_x_expanded = self.sobel_x.repeat(self.n_channels, 1, 1, 1)
        sobel_y_expanded = self.sobel_y.repeat(self.n_channels, 1, 1, 1)
        
        dx = F.conv2d(x, sobel_x_expanded, groups=self.n_channels, padding=1)
        dy = F.conv2d(x, sobel_y_expanded, groups=self.n_channels, padding=1)
        
        return torch.cat([x, dx, dy], dim=1)
    
    def forward(self, x, growth_boost=1.0):
        """Clean forward pass without in-place operations"""
        # Perception
        perceived = self.perceive(x)
        
        # Update
        dx = self.update_net(perceived)
        
        # Apply growth rate with boost
        dx = dx * self.growth_rate * growth_boost
        
        # Growth encouragement for low populations
        alive_mask = (x[:, 3:4] > self.alive_threshold).float()
        alive_ratio = alive_mask.mean()
        
        if alive_ratio < 0.2:
            # Emergency growth boost
            neighbor_alive = F.avg_pool2d(alive_mask, 3, stride=1, padding=1)
            growth_zones = (neighbor_alive > 0.01).float()
            
            # Create growth boost tensor
            growth_boost_tensor = torch.zeros_like(dx)
            growth_boost_tensor[:, 3:4] = growth_zones * 0.2  # Alpha boost
            
            dx = dx + growth_boost_tensor
        
        # Update state
        new_x = x + dx
        
        # Simple bounds - avoid in-place operations
        new_x = torch.clamp(new_x, -1, 1)
        alpha_clamped = torch.clamp(new_x[:, 3:4], 0, 1)  # Alpha in [0,1]
        new_x = torch.cat([new_x[:, :3], alpha_clamped, new_x[:, 4:]], dim=1)
        
        return new_x

def simple_seed(size=32):
    """Simple seed initialization"""
    state = torch.zeros(1, 8, size, size)
    center = size // 2
    
    # Central cross
    state[0, 3, center-2:center+3, center-2:center+3] = 1.0  # Alpha
    state[0, 0, center, center] = 1.0  # Red center
    
    return state

def simple_target(size=32):
    """Simple circular target"""
    target = torch.zeros(1, 8, size, size)
    center = size // 2
    radius = size // 4
    
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    mask = ((x - center) ** 2 + (y - center) ** 2) < radius ** 2
    
    target[0, 0, mask] = 1.0  # Red
    target[0, 3, mask] = 1.0  # Alpha
    
    return target

def train_simple_growth():
    """Simple growth training"""
    print("🌱 Simple Growth NCA Training")
    print("=" * 40)
    
    device = torch.device('cpu')
    size = 32
    
    # Initialize
    model = SimpleGrowthNCA(8).to(device)
    target = simple_target(size)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    target_alive = (target[:, 3:4] > 0.1).float().mean().item()
    print(f"🎯 Target alive ratio: {target_alive:.3f}")
    
    epochs = 100
    results = []
    
    print(f"\n🚀 Training for Growth")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Initialize
        state = simple_seed(size)
        
        # Run CA
        for step in range(15):
            # Progressive growth boost - adjusted for 100 epochs
            if epoch < 30:
                boost = 2.0  # Strong early growth
            elif epoch < 60:
                boost = 1.5  # Moderate growth
            elif epoch < 80:
                boost = 1.2  # Light boost
            else:
                boost = 1.0  # Normal growth
            state = model(state, growth_boost=boost)
        
        # Loss calculation
        alive_mask = (state[:, 3:4] > 0.1).float()
        current_alive = alive_mask.mean()
        
        # Simple losses
        alive_loss = F.mse_loss(current_alive, torch.tensor(target_alive * 0.8))  # Aim for 80% of target
        pattern_loss = F.mse_loss(state[:, :4], target[:, :4])
        
        # Combined loss - prioritize growth
        total_loss = 0.7 * alive_loss + 0.3 * pattern_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record
        results.append({
            "epoch": epoch,
            "alive_ratio": current_alive.item(),
            "alive_loss": alive_loss.item(),
            "pattern_loss": pattern_loss.item(),
            "total_loss": total_loss.item()
        })
        
        # Progress - show every 10 epochs for 100 epoch run
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}: Alive={current_alive.item():.4f}, "
                  f"Loss={total_loss.item():.4f}")
        
        # Success check
        if current_alive.item() > 0.15:
            print(f"   🎉 Growth success! Alive ratio: {current_alive.item():.3f}")
    
    training_time = time.time() - start_time
    
    # Final test
    print(f"\n🎯 Final Test")
    model.eval()
    
    with torch.no_grad():
        test_state = simple_seed(size)
        test_history = [test_state.clone()]
        
        for step in range(30):
            test_state = model(test_state, growth_boost=1.0)
            if step % 5 == 0:
                test_history.append(test_state.clone())
        
        final_alive = (test_state[:, 3:4] > 0.1).float().mean().item()
        final_pattern = F.mse_loss(test_state[:, :4], target[:, :4]).item()
    
    # Results
    alive_ratios = [r["alive_ratio"] for r in results]
    max_alive = max(alive_ratios)
    final_training_alive = alive_ratios[-1]
    
    print(f"   📊 Results:")
    print(f"      Max training alive: {max_alive:.4f}")
    print(f"      Final training alive: {final_training_alive:.4f}")
    print(f"      Final test alive: {final_alive:.4f}")
    print(f"      Pattern loss: {final_pattern:.4f}")
    print(f"      Training time: {training_time:.1f}s")
    
    # Assessment
    growth_success = max_alive > 0.12
    sustained_growth = final_alive > 0.08
    
    print(f"\n🏆 Assessment:")
    print(f"   🌱 Growth Success: {'✅' if growth_success else '❌'} (max: {max_alive:.3f})")
    print(f"   🔄 Sustained Growth: {'✅' if sustained_growth else '❌'} (final: {final_alive:.3f})")
    
    if growth_success and sustained_growth:
        print(f"\n🎉 SUCCESS! Simple approach achieved stable growth!")
    elif growth_success:
        print(f"\n🔄 PARTIAL: Achieved growth but sustainability unclear")
    else:
        print(f"\n❌ CHALLENGE: Still struggling with basic growth")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        "approach": "Simple Growth",
        "results": results,
        "summary": {
            "max_alive": max_alive,
            "final_training_alive": final_training_alive,
            "final_test_alive": final_alive,
            "final_pattern_loss": final_pattern,
            "training_time": training_time,
            "growth_success": bool(growth_success),
            "sustained_growth": bool(sustained_growth)
        }
    }
    
    with open(f'simple_growth_results_{timestamp}.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n💾 Results saved to simple_growth_results_{timestamp}.json")
    
    return save_data, test_history

if __name__ == "__main__":
    print("🧬 Simple Growth NCA Experiment")
    print("   Goal: Clean implementation focused purely on growth")
    print("   Strategy: Avoid complexity, focus on basic growth dynamics")
    print()
    
    results, history = train_simple_growth() 