import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated_fast import IntegratedNCA

def test_fine_tuned_configs():
    """Test fine-tuned NCA configurations to find the sweet spot"""
    device = torch.device("cpu")
    
    # Fine-tuned configurations targeting the sweet spot between death and overgrowth
    configs = {
        "Sweet1": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.1) * 0.32 + 0.73,
            "description": "Sweet spot attempt 1"
        },
        "Sweet2": {
            "alive_threshold": 0.035,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.15) * 0.28 + 0.76,
            "description": "Sweet spot attempt 2"
        },
        "Sweet3": {
            "alive_threshold": 0.045,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.05) * 0.35 + 0.71,
            "description": "Sweet spot attempt 3"
        },
        "Plateau": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.clamp(0.75 + (ns - 1.0) * 0.05, 0.7, 0.95),
            "description": "Linear plateau approach"
        },
        "Stepped": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.where(
                ns < 0.5, torch.ones_like(ns) * 0.6,
                torch.where(ns < 1.5, torch.ones_like(ns) * 0.85,
                torch.where(ns < 3.0, torch.ones_like(ns) * 0.95,
                torch.ones_like(ns) * 0.75))
            ),
            "description": "Stepped survival rates"
        },
        "Gaussian": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: 0.75 + 0.2 * torch.exp(-((ns - 1.5) ** 2) / 1.0),
            "description": "Gaussian peak at 1.5 neighbors"
        },
        "Oscillating": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: 0.8 + 0.1 * torch.sin(ns * 2.0) * torch.exp(-ns * 0.2),
            "description": "Oscillating with decay"
        },
        "Threshold": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.where(
                (ns >= 1.0) & (ns <= 2.5), torch.ones_like(ns) * 0.92,
                torch.ones_like(ns) * 0.78
            ),
            "description": "Binary threshold 1-2.5 neighbors"
        }
    }
    
    results = {}
    
    print("🎯 Fine-tuning NCA configurations for the sweet spot...")
    print("=" * 70)
    
    for config_name, config in configs.items():
        print(f"\n🔬 Testing {config_name}: {config['description']}")
        
        # Test this configuration
        result = test_config_with_custom_growth(config, device)
        results[config_name] = result
    
    # Print detailed summary
    print("\n" + "=" * 70)
    print("📊 DETAILED RESULTS:")
    print("=" * 70)
    
    healthy_configs = []
    for config_name, result in results.items():
        status_emoji = {
            "HEALTHY": "✅",
            "OVERGROWN": "⚠️ ",
            "DEAD": "❌",
            "DYING": "⚠️ ",
            "TOO_DENSE": "⚠️ "
        }.get(result["status"], "❓")
        
        print(f"{status_emoji} {config_name:12s}: {result['initial_alive']:3.0f} → {result['final_alive']:4.0f} cells "
              f"({result['coverage']:5.1f}%) - {result['status']}")
        
        if result["status"] == "HEALTHY":
            healthy_configs.append((config_name, result))
    
    if healthy_configs:
        print(f"\n🏆 FOUND {len(healthy_configs)} HEALTHY CONFIG(S)!")
        best_config = max(healthy_configs, key=lambda x: x[1]["final_alive"])
        print(f"🥇 BEST: {best_config[0]} with {best_config[1]['final_alive']:.0f} alive cells")
        
        # Apply the best configuration to the main training file
        apply_best_config(best_config[0], configs[best_config[0]])
    else:
        print(f"\n😞 NO HEALTHY CONFIGS FOUND")
        # Find the closest to healthy
        best_attempt = min(results.items(), 
                          key=lambda x: abs(x[1]["final_alive"] - 500))  # Target ~500 cells
        print(f"🤔 CLOSEST ATTEMPT: {best_attempt[0]} with {best_attempt[1]['final_alive']:.0f} cells")
    
    # Create visualization
    create_fine_tuned_plot(results)
    
    return results, healthy_configs

def test_config_with_custom_growth(config, device):
    """Test a configuration with custom growth dynamics"""
    # Create NCA instance
    nca = IntegratedNCA(channel_n=8, w_dim=128, hidden_n=64)
    nca.eval()
    
    # Create test seed
    batch_size = 1
    img_size = 64
    seed = nca.get_seed(batch_size, img_size, device, seed_type="distributed")
    
    initial_alive = (seed[0, 3] > config["alive_threshold"]).sum().item()
    
    # Create dummy w vector
    w = torch.randn(batch_size, 128, device=device)
    
    # Test growth with custom dynamics
    x = seed.clone()
    alive_counts = []
    
    # Simulate NCA steps with custom growth
    for step_group in range(7):  # 0, 5, 10, 15, 20, 25, 30
        if step_group > 0:
            # Run 5 NCA steps with custom growth
            for _ in range(5):
                # Basic NCA update (simplified)
                perceived = nca.perceive(x) * nca.perception_scale
                hidden = nca.perception_net(perceived)
                
                # Style modulation (simplified)
                s1 = nca.style_mod1(w)
                s1_scale, s1_bias = s1.chunk(2, dim=1)
                s1_scale = s1_scale.view(1, -1, 1, 1)
                s1_bias = s1_bias.view(1, -1, 1, 1)
                hidden = hidden * (s1_scale + 1) + s1_bias
                
                # Update
                w_spatial = w.view(1, nca.w_dim, 1, 1).expand(1, nca.w_dim, img_size, img_size)
                hidden_combined = torch.cat([hidden, w_spatial], dim=1)
                dx = nca.update_net(hidden_combined) * nca.update_scale
                
                s2 = nca.style_mod2(w)
                s2_scale, s2_bias = s2.chunk(2, dim=1)
                s2_scale = s2_scale.view(1, -1, 1, 1)
                s2_bias = s2_bias.view(1, -1, 1, 1)
                dx = dx * (s2_scale + 1) + s2_bias
                
                x = x + dx
                
                # CUSTOM GROWTH DYNAMICS
                alive_mask = (x[:, 3:4, :, :] > config["alive_threshold"]).float()
                neighbor_sum = F.conv2d(alive_mask, nca.growth_kernel, padding=1)
                growth_factor = config["growth_formula"](neighbor_sum)
                x = x * growth_factor
                
                # Clamping
                x.clamp_(-1.0, 1.0)
        
        # Count alive cells
        alive_mask = (x[0, 3] > config["alive_threshold"]).float()
        alive_count = alive_mask.sum().item()
        alive_counts.append(alive_count)
        
        # Early termination if dead
        if alive_count == 0:
            break
    
    final_alive = alive_counts[-1]
    coverage = final_alive / 4096 * 100
    
    # Determine status
    if final_alive == 0:
        status = "DEAD"
    elif final_alive == 4096:
        status = "OVERGROWN"
    elif final_alive < 50:
        status = "DYING"
    elif final_alive > 2000:
        status = "TOO_DENSE"
    elif 50 <= final_alive <= 2000:
        status = "HEALTHY"
    else:
        status = "UNKNOWN"
    
    return {
        "initial_alive": initial_alive,
        "final_alive": final_alive,
        "coverage": coverage,
        "status": status,
        "alive_counts": alive_counts,
        "final_state": x.clone()
    }

def apply_best_config(config_name, config):
    """Apply the best configuration to the training file"""
    print(f"\n🔧 Applying {config_name} configuration to train_integrated_fast.py...")
    
    # Read the current file
    with open('train_integrated_fast.py', 'r') as f:
        content = f.read()
    
    # Extract the growth formula as a string representation
    threshold = config["alive_threshold"]
    
    # Create the replacement code based on the config
    if config_name == "Sweet1":
        new_growth_code = f'''            # Applied Sweet1 configuration - optimal balance found
            alive_mask = (x[:, 3:4, :, :] > {threshold}).float()
            neighbor_sum = F.conv2d(alive_mask, self.growth_kernel, padding=1)
            growth_factor = torch.sigmoid(neighbor_sum - 1.1) * 0.32 + 0.73
            x = x * growth_factor'''
    elif config_name == "Threshold":
        new_growth_code = f'''            # Applied Threshold configuration - binary survival zones
            alive_mask = (x[:, 3:4, :, :] > {threshold}).float()
            neighbor_sum = F.conv2d(alive_mask, self.growth_kernel, padding=1)
            growth_factor = torch.where(
                (neighbor_sum >= 1.0) & (neighbor_sum <= 2.5), 
                torch.ones_like(neighbor_sum) * 0.92,
                torch.ones_like(neighbor_sum) * 0.78
            )
            x = x * growth_factor'''
    else:
        print(f"⚠️  Configuration {config_name} not implemented for auto-apply")
        return
    
    # Find and replace the growth dynamics section
    import re
    pattern = r'            # .*?growth dynamics.*?\n.*?x = x \* growth_factor'
    
    new_content = re.sub(pattern, new_growth_code, content, flags=re.DOTALL)
    
    # Write back to file
    with open('train_integrated_fast.py', 'w') as f:
        f.write(new_content)
    
    print(f"✅ Applied {config_name} configuration successfully!")

def create_fine_tuned_plot(results):
    """Create visualization of fine-tuned results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Growth curves
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (config_name, result) in enumerate(results.items()):
        steps = list(range(0, len(result["alive_counts"]) * 5, 5))
        color = colors[i]
        
        ax.plot(steps, result["alive_counts"], 
               color=color, label=config_name, 
               linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Alive Cells')
    ax.set_title('Fine-Tuned Growth Curves')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4200)
    
    # Coverage comparison
    ax = axes[0, 1]
    names = list(results.keys())
    coverages = [results[name]["coverage"] for name in names]
    colors_bar = colors[:len(names)]
    
    bars = ax.bar(names, coverages, color=colors_bar, alpha=0.7)
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Final Coverage')
    ax.set_ylim(0, 105)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add status on bars
    for bar, name in zip(bars, names):
        status = results[name]["status"]
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                status, ha='center', va='bottom', fontsize=7)
    
    # Show best 2 final states
    healthy_results = [(name, res) for name, res in results.items() if res["status"] == "HEALTHY"]
    if not healthy_results:
        # Show 2 closest to healthy
        sorted_results = sorted(results.items(), key=lambda x: abs(x[1]["final_alive"] - 500))
        healthy_results = sorted_results[:2]
    
    for i, (config_name, result) in enumerate(healthy_results[:2]):
        ax = axes[1, i]
        alpha = result["final_state"][0, 3].detach().numpy()
        im = ax.imshow(alpha, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'{config_name}\n{result["final_alive"]:.0f} cells ({result["status"]})')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('fine_tuned_nca_results.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved fine-tuned results as 'fine_tuned_nca_results.png'")

if __name__ == "__main__":
    results, healthy = test_fine_tuned_configs()
    print(f"\n✅ Fine-tuning complete! Found {len(healthy)} healthy configurations.") 