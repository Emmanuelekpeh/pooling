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

class FastStableNCA(nn.Module):
    """Optimized version with reduced complexity for faster training"""
    def __init__(self, n_channels=8, hidden_size=32):
        super().__init__()
        self.n_channels = n_channels
        
        # Simplified perception - just Sobel + identity
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Small network
        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 3, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, n_channels, 1),
            nn.Tanh()
        )
        
        # Learnable stability parameters
        self.stability_factor = nn.Parameter(torch.tensor(0.1))
        
        # Initialize small
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def perceive(self, x):
        # Expand kernels to match number of channels
        sobel_x_expanded = self.sobel_x.repeat(self.n_channels, 1, 1, 1)
        sobel_y_expanded = self.sobel_y.repeat(self.n_channels, 1, 1, 1)
        
        dx = F.conv2d(x, sobel_x_expanded, groups=self.n_channels, padding=1)
        dy = F.conv2d(x, sobel_y_expanded, groups=self.n_channels, padding=1)
        return torch.cat([x, dx, dy], dim=1)
    
    def forward(self, x, target_alive_ratio=0.1):
        perceived = self.perceive(x)
        dx = self.update_net(perceived) * self.stability_factor
        
        # Stability control
        alive_mask = (x[:, 3:4] > 0.1).float()
        current_alive = alive_mask.mean()
        
        # Scale update based on population
        if current_alive < target_alive_ratio * 0.8:
            scale = 1.2  # Encourage growth
        elif current_alive > target_alive_ratio * 1.2:
            scale = 0.8  # Discourage growth
        else:
            scale = 1.0
        
        new_x = x + dx * scale
        new_x = new_x * alive_mask  # Dead cells stay dead
        new_x = torch.clamp(new_x, -1, 1)
        
        return new_x

def create_simple_target(size=32):
    target = torch.zeros(1, 8, size, size)
    center = size // 2
    radius = size // 4
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    mask = ((x - center) ** 2 + (y - center) ** 2) < radius ** 2
    target[0, 0, mask] = 1.0  # Red
    target[0, 3, mask] = 1.0  # Alpha
    return target

def initialize_seed(size=32):
    state = torch.zeros(1, 8, size, size)
    center = size // 2
    state[0, 3, center-1:center+2, center-1:center+2] = 1.0  # Alpha
    state[0, 0, center, center] = 1.0  # Red center
    return state

def train_fast():
    print("🚀 Fast Stable NCA Training")
    print("=" * 40)
    
    device = torch.device('cpu')
    size = 32
    
    model = FastStableNCA(8, 32).to(device)
    target = create_simple_target(size)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # 5 curriculum stages - much faster
    stages = [
        {"name": "Stability", "target": 0.05, "epochs": 30},
        {"name": "Growth", "target": 0.1, "epochs": 30},
        {"name": "Control", "target": 0.15, "epochs": 30},
        {"name": "Pattern", "target": 0.2, "epochs": 30},
        {"name": "Mastery", "target": 0.25, "epochs": 30}
    ]
    
    results = []
    start_time = time.time()
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n📚 Stage {stage_idx + 1}: {stage['name']} (target: {stage['target']})")
        
        stage_start = time.time()
        losses = []
        alive_ratios = []
        
        for epoch in range(stage['epochs']):
            model.train()
            optimizer.zero_grad()
            
            state = initialize_seed(size)
            
            # Run for fewer steps
            for step in range(15):
                state = model(state, target_alive_ratio=stage['target'])
            
            # Loss calculation
            pattern_loss = F.mse_loss(state[:, :4], target[:, :4])
            alive_mask = (state[:, 3:4] > 0.1).float()
            current_alive = alive_mask.mean()
            stability_loss = F.mse_loss(current_alive, torch.tensor(stage['target']))
            
            total_loss = pattern_loss + stability_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(total_loss.item())
            alive_ratios.append(current_alive.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:2d}: Loss={total_loss.item():.4f}, Alive={current_alive.item():.3f}")
        
        stage_time = time.time() - stage_start
        avg_loss = np.mean(losses[-5:])
        avg_alive = np.mean(alive_ratios[-5:])
        converged = abs(avg_alive - stage['target']) < 0.03
        
                 results.append({
             "stage": stage['name'],
             "target": stage['target'],
             "final_alive": avg_alive,
             "final_loss": avg_loss,
             "time": stage_time,
             "converged": bool(converged)  # Ensure JSON serializable
         })
        
        print(f"  ✅ Completed in {stage_time:.1f}s - Loss: {avg_loss:.4f}, Alive: {avg_alive:.3f}, Converged: {converged}")
    
    total_time = time.time() - start_time
    
    # Final test
    print(f"\n🎯 Final Test")
    model.eval()
    with torch.no_grad():
        test_state = initialize_seed(size)
        test_states = [test_state.clone()]
        
        for step in range(30):
            test_state = model(test_state, target_alive_ratio=0.2)
            if step % 5 == 0:
                test_states.append(test_state.clone())
        
        final_alive = (test_state[:, 3:4] > 0.1).float().mean().item()
        final_loss = F.mse_loss(test_state[:, :4], target[:, :4]).item()
    
    print(f"Final alive ratio: {final_alive:.3f}")
    print(f"Final pattern loss: {final_loss:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"All stages converged: {all(r['converged'] for r in results)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = {
        "stages": results,
        "final_alive": final_alive,
        "final_loss": final_loss,
        "total_time": total_time,
        "success": all(r['converged'] for r in results)
    }
    
    with open(f'fast_nca_results_{timestamp}.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n📊 Results saved to fast_nca_results_{timestamp}.json")
    
    return result_data, test_states

if __name__ == "__main__":
    results, states = train_fast() 