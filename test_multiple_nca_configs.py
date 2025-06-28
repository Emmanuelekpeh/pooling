import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated_fast import IntegratedNCA
import copy

def test_multiple_nca_configs():
    """Test multiple NCA configurations simultaneously"""
    device = torch.device("cpu")
    
    # Define different growth configurations to test
    configs = {
        "Original": {
            "alive_threshold": 0.01,
            "growth_formula": lambda ns: torch.sigmoid(ns - 0.5) * 0.4 + 0.6,
            "description": "Original overgrowth formula"
        },
        "Conservative": {
            "alive_threshold": 0.03,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.2) * 0.3 + 0.75,
            "description": "Conservative growth (current)"
        },
        "Moderate": {
            "alive_threshold": 0.05,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.0) * 0.35 + 0.7,
            "description": "Moderate growth control"
        },
        "Aggressive": {
            "alive_threshold": 0.08,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.5) * 0.4 + 0.65,
            "description": "Aggressive death pressure"
        },
        "Conway-like": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.where(
                (ns >= 2) & (ns <= 3), torch.ones_like(ns) * 0.95,
                torch.where(ns == 3, torch.ones_like(ns) * 1.02, torch.ones_like(ns) * 0.8)
            ),
            "description": "Conway Game of Life inspired"
        },
        "Balanced": {
            "alive_threshold": 0.06,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.3) * 0.25 + 0.78,
            "description": "Balanced survival/death"
        }
    }
    
    results = {}
    
    print("🧬 Testing multiple NCA configurations...")
    print("=" * 60)
    
    for config_name, config in configs.items():
        print(f"\n🔬 Testing {config_name}: {config['description']}")
        
        # Create NCA instance
        nca = IntegratedNCA(channel_n=8, w_dim=128, hidden_n=64)
        nca.eval()
        
        # Override the growth dynamics temporarily
        original_forward = nca.forward
        
        def custom_forward(x, w, steps, target_img=None):
            batch_size, _, height, width = x.shape
            
            # Pre-compute style modulations (same as original)
            s1 = nca.style_mod1(w)
            s1_scale, s1_bias = s1.chunk(2, dim=1)
            s1_scale = s1_scale.view(batch_size, -1, 1, 1)
            s1_bias = s1_bias.view(batch_size, -1, 1, 1)
            
            s2 = nca.style_mod2(w)
            s2_scale, s2_bias = s2.chunk(2, dim=1)
            s2_scale = s2_scale.view(batch_size, -1, 1, 1)
            s2_bias = s2_bias.view(batch_size, -1, 1, 1)
            
            w_spatial = w.view(batch_size, nca.w_dim, 1, 1).expand(batch_size, nca.w_dim, height, width)
            
            for step in range(steps):
                # Perception and update (same as original)
                perceived = nca.perceive(x) * nca.perception_scale
                hidden = nca.perception_net(perceived)
                hidden = hidden * (s1_scale + 1) + s1_bias
                
                hidden_combined = torch.cat([hidden, w_spatial], dim=1)
                dx = nca.update_net(hidden_combined) * nca.update_scale
                dx = dx * (s2_scale + 1) + s2_bias
                
                x = x + dx
                
                # CUSTOM GROWTH DYNAMICS FOR THIS CONFIG
                alive_mask = (x[:, 3:4, :, :] > config["alive_threshold"]).float()
                neighbor_sum = F.conv2d(alive_mask, nca.growth_kernel, padding=1)
                growth_factor = config["growth_formula"](neighbor_sum)
                x = x * growth_factor
                
                # Noise and clamping
                if nca.training:
                    x.add_(torch.randn_like(x), alpha=nca.noise_scale.item())
                x.clamp_(-1.0, 1.0)
            
            return x
        
        # Temporarily replace forward method
        nca.forward = custom_forward
        
        # Test this configuration
        result = test_single_config(nca, config_name, config, device)
        results[config_name] = result
        
        # Restore original forward method
        nca.forward = original_forward
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY OF ALL CONFIGURATIONS:")
    print("=" * 60)
    
    for config_name, result in results.items():
        status_emoji = {
            "HEALTHY": "✅",
            "OVERGROWN": "⚠️ ",
            "DEAD": "❌",
            "DYING": "⚠️ ",
            "TOO_DENSE": "⚠️ "
        }.get(result["status"], "❓")
        
        print(f"{status_emoji} {config_name:12s}: {result['final_alive']:4.0f} cells ({result['coverage']:5.1f}%) - {result['status']}")
    
    # Find best configuration
    healthy_configs = {name: res for name, res in results.items() 
                      if res["status"] == "HEALTHY"}
    
    if healthy_configs:
        best_config = max(healthy_configs.items(), 
                         key=lambda x: x[1]["final_alive"])
        print(f"\n🏆 BEST CONFIG: {best_config[0]} with {best_config[1]['final_alive']:.0f} alive cells")
    else:
        print(f"\n⚠️  NO HEALTHY CONFIGS FOUND - All either died or overgrew")
    
    # Create comparison visualization
    create_comparison_plot(results)
    
    return results

def test_single_config(nca, config_name, config, device):
    """Test a single NCA configuration"""
    # Create test seed
    batch_size = 1
    img_size = 64
    seed = nca.get_seed(batch_size, img_size, device, seed_type="distributed")
    
    initial_alive = (seed[0, 3] > config["alive_threshold"]).sum().item()
    
    # Create dummy w vector
    w = torch.randn(batch_size, 128, device=device)
    
    # Test growth over steps
    x = seed.clone()
    alive_counts = []
    
    for step in range(0, 31, 5):
        if step > 0:
            x = nca(x, w, steps=5)
        
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
    
    print(f"  Initial: {initial_alive:3.0f} → Final: {final_alive:4.0f} cells ({coverage:5.1f}%) - {status}")
    
    return {
        "config_name": config_name,
        "initial_alive": initial_alive,
        "final_alive": final_alive,
        "coverage": coverage,
        "status": status,
        "alive_counts": alive_counts,
        "final_state": x.clone()
    }

def create_comparison_plot(results):
    """Create a comparison plot of all configurations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    # Plot 1: Growth curves
    ax = axes[0]
    for i, (config_name, result) in enumerate(results.items()):
        steps = list(range(0, len(result["alive_counts"]) * 5, 5))
        ax.plot(steps, result["alive_counts"], 
               color=colors[i % len(colors)], 
               label=config_name, 
               linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Alive Cells')
    ax.set_title('Growth Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4200)
    
    # Plot 2: Final coverage bar chart
    ax = axes[1]
    names = list(results.keys())
    coverages = [results[name]["coverage"] for name in names]
    colors_bar = [colors[i % len(colors)] for i in range(len(names))]
    
    bars = ax.bar(names, coverages, color=colors_bar, alpha=0.7)
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Final Coverage Comparison')
    ax.set_ylim(0, 105)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add status labels on bars
    for bar, name in zip(bars, names):
        status = results[name]["status"]
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                status, ha='center', va='bottom', fontsize=8)
    
    # Plots 3-6: Show final states of first 4 configs
    for i, (config_name, result) in enumerate(list(results.items())[:4]):
        ax = axes[i + 2]
        
        # Show alpha channel
        alpha = result["final_state"][0, 3].detach().numpy()
        im = ax.imshow(alpha, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'{config_name}\n{result["final_alive"]:.0f} cells ({result["status"]})')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('nca_configs_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved comparison plot as 'nca_configs_comparison.png'")

if __name__ == "__main__":
    results = test_multiple_nca_configs()
    print(f"\n✅ Tested {len(results)} configurations!") 