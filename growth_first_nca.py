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

class GrowthFirstNCA(nn.Module):
    """Growth-First NCA with strong growth incentives"""
    def __init__(self, n_channels=8, hidden_size=48):
        super().__init__()
        self.n_channels = n_channels
        
        # Perception kernels
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('laplacian', torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Larger network for better learning capacity
        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 4, hidden_size, 1),  # 4 = original + 3 perceptions
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, n_channels, 1),
            nn.Tanh()
        )
        
        # Growth-focused parameters - much stronger than before
        self.growth_factor = nn.Parameter(torch.tensor(0.5))  # Increased from 0.1
        self.survival_threshold = nn.Parameter(torch.tensor(0.1))
        self.growth_threshold = nn.Parameter(torch.tensor(0.15))
        
        # Initialize for growth
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)  # Larger initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def perceive(self, x):
        """Enhanced perception with more detail"""
        sobel_x_expanded = self.sobel_x.repeat(self.n_channels, 1, 1, 1)
        sobel_y_expanded = self.sobel_y.repeat(self.n_channels, 1, 1, 1)
        laplacian_expanded = self.laplacian.repeat(self.n_channels, 1, 1, 1)
        
        dx = F.conv2d(x, sobel_x_expanded, groups=self.n_channels, padding=1)
        dy = F.conv2d(x, sobel_y_expanded, groups=self.n_channels, padding=1)
        lap = F.conv2d(x, laplacian_expanded, groups=self.n_channels, padding=1)
        
        return torch.cat([x, dx, dy, lap], dim=1)
    
    def forward(self, x, growth_incentive=1.0, stability_weight=0.5):
        """Forward pass with configurable growth incentives"""
        perceived = self.perceive(x)
        dx = self.update_net(perceived)
        
        # Current alive state
        alive_mask = (x[:, 3:4] > self.survival_threshold).float()
        current_alive_ratio = alive_mask.mean()
        
        # Growth dynamics - encourage expansion
        neighbor_alive = F.avg_pool2d(alive_mask, 3, stride=1, padding=1)
        
        # Strong growth incentive for low populations
        if current_alive_ratio < 0.05:
            growth_boost = 3.0 * growth_incentive  # Very strong boost
        elif current_alive_ratio < 0.1:
            growth_boost = 2.0 * growth_incentive  # Strong boost
        elif current_alive_ratio < 0.2:
            growth_boost = 1.5 * growth_incentive  # Moderate boost
        else:
            growth_boost = growth_incentive  # Normal growth
        
        # Apply growth factor with incentive
        scaled_dx = dx * self.growth_factor * growth_boost
        
        # Encourage growth in areas with living neighbors
        growth_mask = (neighbor_alive > 0.1).float()
        growth_encouragement = growth_mask * 0.1 * growth_incentive
        
        # Update state
        new_x = x + scaled_dx
        
        # Add growth encouragement to alpha channel
        new_x[:, 3:4] = new_x[:, 3:4] + growth_encouragement
        
        # Apply stability constraints (lighter in early stages)
        stability_factor = 0.5 + stability_weight * 0.5  # Range: 0.5 to 1.0
        
        # Keep alive cells alive, allow dead cells to potentially come alive
        alive_preservation = alive_mask * stability_factor
        new_x = new_x * (alive_preservation + (1 - alive_mask))
        
        # Bounds
        new_x = torch.clamp(new_x, -1, 1)
        
        return new_x

def create_growth_target(size=32, pattern_type="expanding_circle"):
    """Create targets that encourage growth"""
    target = torch.zeros(1, 8, size, size)
    center = size // 2
    
    if pattern_type == "expanding_circle":
        # Larger circle to encourage more growth
        radius = size // 3  # Increased from size // 4
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        mask = ((x - center) ** 2 + (y - center) ** 2) < radius ** 2
        target[0, 0, mask] = 1.0  # Red
        target[0, 3, mask] = 1.0  # Alpha
        
    elif pattern_type == "cross_pattern":
        # Large cross pattern
        thickness = 3
        target[0, 1, center-thickness:center+thickness, :] = 1.0  # Green horizontal
        target[0, 1, :, center-thickness:center+thickness] = 1.0  # Green vertical
        target[0, 3, center-thickness:center+thickness, :] = 1.0  # Alpha horizontal
        target[0, 3, :, center-thickness:center+thickness] = 1.0  # Alpha vertical
        
    elif pattern_type == "scattered_dots":
        # Multiple growth centers
        positions = [(8, 8), (8, 24), (24, 8), (24, 24), (16, 16)]
        for px, py in positions:
            target[0, 2, py-2:py+3, px-2:px+3] = 1.0  # Blue
            target[0, 3, py-2:py+3, px-2:px+3] = 1.0  # Alpha
    
    return target

def initialize_growth_seed(size=32, seed_type="multiple"):
    """Initialize with seeds designed to encourage growth"""
    state = torch.zeros(1, 8, size, size)
    center = size // 2
    
    if seed_type == "multiple":
        # Multiple small seeds to encourage distributed growth
        positions = [(center-4, center-4), (center+4, center+4), (center, center)]
        for px, py in positions:
            state[0, 3, py-1:py+2, px-1:px+2] = 1.0  # Alpha
            state[0, 0, py, px] = 1.0  # Red center
            
    elif seed_type == "line":
        # Horizontal line seed
        state[0, 3, center, center-3:center+4] = 1.0  # Alpha line
        state[0, 1, center, center] = 1.0  # Green center
        
    elif seed_type == "large_center":
        # Larger central seed
        state[0, 3, center-2:center+3, center-2:center+3] = 1.0  # Alpha
        state[0, 0, center-1:center+2, center-1:center+2] = 0.8  # Red
        
    return state

def growth_first_curriculum():
    """Define Growth-First curriculum stages"""
    return [
        {
            "name": "Aggressive Growth", 
            "target_alive": 0.20,  # Start with substantial target
            "epochs": 40,
            "growth_incentive": 2.5,  # Very strong growth
            "stability_weight": 0.1,  # Minimal stability constraints
            "pattern_weight": 0.3,
            "steps": 25
        },
        {
            "name": "Controlled Growth",
            "target_alive": 0.25,
            "epochs": 40,
            "growth_incentive": 2.0,  # Strong growth
            "stability_weight": 0.2,  # Light stability
            "pattern_weight": 0.5,
            "steps": 30
        },
        {
            "name": "Stable Dynamics",
            "target_alive": 0.30,
            "epochs": 40,
            "growth_incentive": 1.5,  # Moderate growth
            "stability_weight": 0.4,  # Balanced
            "pattern_weight": 0.6,
            "steps": 35
        },
        {
            "name": "Pattern Formation",
            "target_alive": 0.25,  # Slight reduction for pattern focus
            "epochs": 40,
            "growth_incentive": 1.2,
            "stability_weight": 0.5,  # More stability
            "pattern_weight": 0.8,  # Focus on patterns
            "steps": 40
        },
        {
            "name": "Robust Control",
            "target_alive": 0.22,
            "epochs": 40,
            "growth_incentive": 1.0,
            "stability_weight": 0.6,  # High stability
            "pattern_weight": 0.9,  # High pattern fidelity
            "steps": 45
        }
    ]

def train_growth_first():
    """Growth-First curriculum training"""
    print("🌱 Growth-First NCA Training")
    print("=" * 45)
    
    device = torch.device('cpu')
    size = 32
    
    # Initialize model and target
    model = GrowthFirstNCA(8, 48).to(device)
    target = create_growth_target(size, "expanding_circle")
    optimizer = optim.Adam(model.parameters(), lr=0.003)  # Higher learning rate
    
    stages = growth_first_curriculum()
    results = {"stages": [], "training_log": []}
    
    total_start_time = time.time()
    print(f"🎯 Target pattern: Expanding circle (alive ratio ~{(target[:, 3:4] > 0.1).float().mean().item():.3f})")
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n🚀 Stage {stage_idx + 1}: {stage['name']}")
        print(f"   Target: {stage['target_alive']:.2f} | Growth: {stage['growth_incentive']:.1f}x | Stability: {stage['stability_weight']:.1f}")
        
        stage_start = time.time()
        stage_losses = []
        stage_alive_ratios = []
        stage_pattern_losses = []
        
        for epoch in range(stage['epochs']):
            model.train()
            optimizer.zero_grad()
            
            # Initialize with growth-oriented seed
            state = initialize_growth_seed(size, "multiple")
            
            # Run NCA with current stage parameters
            for step in range(stage['steps']):
                state = model(state, 
                            growth_incentive=stage['growth_incentive'],
                            stability_weight=stage['stability_weight'])
            
            # Calculate losses
            pattern_loss = F.mse_loss(state[:, :4], target[:, :4])
            
            alive_mask = (state[:, 3:4] > 0.1).float()
            current_alive = alive_mask.mean()
            
            # Stability loss - encourage target alive ratio
            alive_target = torch.tensor(stage['target_alive'])
            stability_loss = F.mse_loss(current_alive, alive_target)
            
            # Combined loss with stage-specific weighting
            total_loss = (stage['pattern_weight'] * pattern_loss + 
                         (1 - stage['pattern_weight']) * stability_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # Higher gradient clip
            optimizer.step()
            
            # Record metrics
            stage_losses.append(total_loss.item())
            stage_alive_ratios.append(current_alive.item())
            stage_pattern_losses.append(pattern_loss.item())
            
            # Progress reporting
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:2d}: Loss={total_loss.item():.4f}, "
                      f"Alive={current_alive.item():.3f}, "
                      f"Pattern={pattern_loss.item():.4f}")
        
        # Stage analysis
        stage_time = time.time() - stage_start
        final_loss = np.mean(stage_losses[-5:])
        final_alive = np.mean(stage_alive_ratios[-5:])
        final_pattern = np.mean(stage_pattern_losses[-5:])
        
        # Check convergence
        alive_progress = abs(final_alive - stage['target_alive']) < 0.05
        loss_stable = len(stage_losses) > 10 and (stage_losses[-1] - stage_losses[-10]) > -0.01
        converged = alive_progress and final_pattern < 0.5
        
        stage_result = {
            "stage": stage['name'],
            "target_alive": stage['target_alive'],
            "final_alive": final_alive,
            "final_loss": final_loss,
            "final_pattern_loss": final_pattern,
            "time_seconds": stage_time,
            "converged": bool(converged),
            "growth_achieved": final_alive > 0.05,  # Basic growth check
            "alive_progress": abs(final_alive - stage['target_alive'])
        }
        
        results['stages'].append(stage_result)
        
        # Status reporting
        growth_status = "✅" if stage_result['growth_achieved'] else "❌"
        conv_status = "✅" if converged else "🔄"
        
        print(f"   {conv_status} Completed in {stage_time:.1f}s")
        print(f"   {growth_status} Growth: {final_alive:.3f} (target: {stage['target_alive']:.3f})")
        print(f"   📉 Losses: Total={final_loss:.4f}, Pattern={final_pattern:.4f}")
        
        # Early success check
        if final_alive > 0.15 and stage_idx == 0:
            print(f"   🎉 Excellent growth in first stage! NCA is learning to live!")
    
    total_time = time.time() - total_start_time
    
    # Final comprehensive test
    print(f"\n🎯 Final Comprehensive Test")
    model.eval()
    test_results = {}
    
    with torch.no_grad():
        # Test with different seeds
        seed_types = ["multiple", "line", "large_center"]
        for seed_type in seed_types:
            test_state = initialize_growth_seed(size, seed_type)
            test_states = [test_state.clone()]
            
            # Run longer test
            for step in range(50):
                test_state = model(test_state, growth_incentive=1.0, stability_weight=0.5)
                if step % 10 == 0:
                    test_states.append(test_state.clone())
            
            final_alive = (test_state[:, 3:4] > 0.1).float().mean().item()
            final_pattern_loss = F.mse_loss(test_state[:, :4], target[:, :4]).item()
            
            test_results[seed_type] = {
                "final_alive": final_alive,
                "final_pattern_loss": final_pattern_loss,
                "states": test_states
            }
            
            print(f"   {seed_type}: Alive={final_alive:.3f}, Pattern Loss={final_pattern_loss:.4f}")
    
    # Overall assessment
    avg_final_alive = np.mean([r["final_alive"] for r in test_results.values()])
    growth_success = avg_final_alive > 0.1
    pattern_success = all(r["final_pattern_loss"] < 0.8 for r in test_results.values())
    all_stages_grew = all(s["growth_achieved"] for s in results['stages'])
    
    print(f"\n📊 Final Assessment:")
    print(f"   🌱 Growth Success: {'✅' if growth_success else '❌'} (avg alive: {avg_final_alive:.3f})")
    print(f"   🎨 Pattern Success: {'✅' if pattern_success else '❌'}")
    print(f"   📈 All Stages Grew: {'✅' if all_stages_grew else '❌'}")
    print(f"   ⏱️  Total Time: {total_time:.1f}s")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results = {
        "curriculum": "Growth-First",
        "stages": results['stages'],
        "test_results": {k: {**v, "states": None} for k, v in test_results.items()},  # Exclude states from JSON
        "summary": {
            "total_time": total_time,
            "avg_final_alive": avg_final_alive,
            "growth_success": bool(growth_success),
            "pattern_success": bool(pattern_success),
            "all_stages_grew": bool(all_stages_grew),
            "overall_success": bool(growth_success and all_stages_grew)
        }
    }
    
    with open(f'growth_first_results_{timestamp}.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save test states for visualization
    np.save(f'growth_first_states_{timestamp}.npy', 
            {k: torch.stack(v["states"]).numpy() for k, v in test_results.items()})
    
    print(f"\n💾 Results saved to growth_first_results_{timestamp}.json")
    
    if growth_success and all_stages_grew:
        print("🎉 SUCCESS: Growth-First curriculum achieved stable NCA growth!")
        print("   The NCA learned to grow from seeds to stable living patterns!")
    elif growth_success:
        print("🔄 PARTIAL SUCCESS: NCA achieved growth but some stages struggled")
    else:
        print("❌ CHALLENGE: NCA still struggling with growth - need stronger incentives")
    
    return final_results, test_results

if __name__ == "__main__":
    print("🧬 Starting Growth-First NCA Curriculum Training")
    print("   Philosophy: Teach growth first, then add stability constraints")
    print("   Key changes: Stronger growth incentives, larger targets, better seeds")
    print()
    
    results, test_states = train_growth_first() 