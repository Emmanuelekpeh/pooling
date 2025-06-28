import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated_fast import IntegratedNCA

def test_ultra_precise_configs():
    """Test ultra-precise NCA configurations with stabilization mechanisms"""
    device = torch.device("cpu")
    
    # Ultra-precise configurations with tiny parameter variations
    configs = {
        "Micro1": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.12) * 0.31 + 0.74,
            "stabilization": "momentum",
            "description": "Micro-adjustment 1 with momentum"
        },
        "Micro2": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.sigmoid(ns - 1.08) * 0.33 + 0.72,
            "stabilization": "momentum",
            "description": "Micro-adjustment 2 with momentum"
        },
        "Clamped1": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.clamp(torch.sigmoid(ns - 1.1) * 0.4 + 0.7, 0.72, 0.98),
            "stabilization": "hard_clamp",
            "description": "Hard clamped growth"
        },
        "Clamped2": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.clamp(torch.sigmoid(ns - 1.0) * 0.35 + 0.75, 0.74, 0.96),
            "stabilization": "hard_clamp",
            "description": "Tighter clamped growth"
        },
        "Adaptive1": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: 0.85 - 0.1 * torch.tanh((ns - 1.5) * 2.0),
            "stabilization": "adaptive",
            "description": "Adaptive based on density"
        },
        "Adaptive2": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: 0.83 - 0.08 * torch.tanh((ns - 1.2) * 1.5),
            "stabilization": "adaptive",
            "description": "Gentler adaptive"
        },
        "Regulated": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: 0.8 + 0.1 / (1 + torch.exp(2 * (ns - 1.5))),
            "stabilization": "regulation",
            "description": "Self-regulating sigmoid"
        },
        "Constrained": {
            "alive_threshold": 0.04,
            "growth_formula": lambda ns: torch.where(
                ns > 3.0, torch.ones_like(ns) * 0.7,  # Strong death for overcrowding
                torch.where(ns < 0.5, torch.ones_like(ns) * 0.65,  # Death for isolation
                0.86 + 0.06 * torch.sin(ns * 1.5))  # Gentle oscillation in middle
            ),
            "stabilization": "constraint",
            "description": "Constrained with oscillation"
        }
    }
    
    results = {}
    
    print("🔬 Ultra-precise NCA parameter search with stabilization...")
    print("=" * 80)
    
    for config_name, config in configs.items():
        print(f"\n🎯 Testing {config_name}: {config['description']}")
        
        # Test this configuration with stabilization
        result = test_stabilized_config(config, device)
        results[config_name] = result
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print("📊 ULTRA-PRECISE RESULTS:")
    print("=" * 80)
    
    healthy_configs = []
    for config_name, result in results.items():
        status_emoji = {
            "HEALTHY": "✅",
            "OVERGROWN": "⚠️ ",
            "DEAD": "❌",
            "DYING": "⚠️ ",
            "TOO_DENSE": "⚠️ ",
            "STABLE": "🟢"
        }.get(result["status"], "❓")
        
        stability_score = result.get("stability_score", 0)
        print(f"{status_emoji} {config_name:12s}: {result['initial_alive']:3.0f} → {result['final_alive']:4.0f} cells "
              f"({result['coverage']:5.1f}%) - {result['status']} (stability: {stability_score:.2f})")
        
        if result["status"] in ["HEALTHY", "STABLE"]:
            healthy_configs.append((config_name, result))
    
    if healthy_configs:
        print(f"\n🏆 FOUND {len(healthy_configs)} STABLE CONFIG(S)!")
        best_config = max(healthy_configs, key=lambda x: x[1].get("stability_score", 0))
        print(f"🥇 MOST STABLE: {best_config[0]} with stability {best_config[1]['stability_score']:.3f}")
        
        # Apply the best configuration
        apply_ultra_precise_config(best_config[0], configs[best_config[0]])
    else:
        print(f"\n😞 NO STABLE CONFIGS FOUND")
        # Show the one with best stability score
        best_attempt = max(results.items(), key=lambda x: x[1].get("stability_score", 0))
        print(f"🤔 MOST STABLE ATTEMPT: {best_attempt[0]} (stability: {best_attempt[1]['stability_score']:.3f})")
    
    # Create ultra-precise visualization
    create_ultra_precise_plot(results)
    
    return results, healthy_configs

def test_stabilized_config(config, device):
    """Test a configuration with stabilization mechanisms"""
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
    
    # Test growth with stabilization
    x = seed.clone()
    alive_counts = []
    momentum = torch.zeros_like(x) if config.get("stabilization") == "momentum" else None
    
    # Simulate NCA steps with stabilization
    for step_group in range(10):  # More steps to test stability
        if step_group > 0:
            # Run 3 NCA steps with stabilization
            for _ in range(3):
                # Basic NCA update
                perceived = nca.perceive(x) * nca.perception_scale
                hidden = nca.perception_net(perceived)
                
                # Style modulation
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
                
                # Apply momentum stabilization
                if config.get("stabilization") == "momentum":
                    momentum = 0.9 * momentum + 0.1 * dx
                    dx = momentum
                
                x = x + dx * 0.5  # Smaller update steps for stability
                
                # CUSTOM GROWTH DYNAMICS WITH STABILIZATION
                alive_mask = (x[:, 3:4, :, :] > config["alive_threshold"]).float()
                neighbor_sum = F.conv2d(alive_mask, nca.growth_kernel, padding=1)
                growth_factor = config["growth_formula"](neighbor_sum)
                
                # Apply stabilization to growth factor
                if config.get("stabilization") == "hard_clamp":
                    growth_factor = torch.clamp(growth_factor, 0.65, 1.05)
                elif config.get("stabilization") == "adaptive":
                    # Adaptive based on current alive percentage
                    alive_ratio = alive_mask.sum() / (64 * 64)
                    if alive_ratio > 0.3:  # Too many alive cells
                        growth_factor = growth_factor * 0.9
                    elif alive_ratio < 0.01:  # Too few alive cells
                        growth_factor = growth_factor * 1.1
                elif config.get("stabilization") == "regulation":
                    # Self-regulation based on growth rate
                    growth_factor = torch.clamp(growth_factor, 0.7, 1.0)
                elif config.get("stabilization") == "constraint":
                    # Constrain extreme values
                    growth_factor = torch.clamp(growth_factor, 0.6, 1.1)
                
                x = x * growth_factor
                
                # Gentle clamping for stability
                x.clamp_(-0.8, 0.8)
        
        # Count alive cells
        alive_mask = (x[0, 3] > config["alive_threshold"]).float()
        alive_count = alive_mask.sum().item()
        alive_counts.append(alive_count)
        
        # Early termination if dead
        if alive_count == 0:
            break
    
    final_alive = alive_counts[-1]
    coverage = final_alive / 4096 * 100
    
    # Calculate stability score
    if len(alive_counts) > 3:
        # Measure how stable the alive count is over time
        recent_counts = alive_counts[-5:]
        stability_score = 1.0 - (np.std(recent_counts) / (np.mean(recent_counts) + 1e-6))
        stability_score = max(0, stability_score)
    else:
        stability_score = 0.0
    
    # Determine status with stability consideration
    if final_alive == 0:
        status = "DEAD"
    elif final_alive == 4096:
        status = "OVERGROWN"
    elif final_alive < 50:
        status = "DYING"
    elif final_alive > 2000:
        status = "TOO_DENSE"
    elif 50 <= final_alive <= 2000:
        if stability_score > 0.8:
            status = "STABLE"
        else:
            status = "HEALTHY"
    else:
        status = "UNKNOWN"
    
    return {
        "initial_alive": initial_alive,
        "final_alive": final_alive,
        "coverage": coverage,
        "status": status,
        "stability_score": stability_score,
        "alive_counts": alive_counts,
        "final_state": x.clone()
    }

def apply_ultra_precise_config(config_name, config):
    """Apply the ultra-precise configuration to the training file"""
    print(f"\n🔧 Applying {config_name} configuration to train_integrated_fast.py...")
    
    # Read the current file
    with open('train_integrated_fast.py', 'r') as f:
        content = f.read()
    
    threshold = config["alive_threshold"]
    stabilization = config.get("stabilization", "none")
    
    # Create the replacement code based on the config
    if config_name == "Micro1":
        new_growth_code = f'''            # Applied Micro1 configuration - ultra-precise with momentum stabilization
            alive_mask = (x[:, 3:4, :, :] > {threshold}).float()
            neighbor_sum = F.conv2d(alive_mask, self.growth_kernel, padding=1)
            growth_factor = torch.sigmoid(neighbor_sum - 1.12) * 0.31 + 0.74
            growth_factor = torch.clamp(growth_factor, 0.65, 1.05)  # Stabilization
            x = x * growth_factor'''
    elif config_name == "Clamped2":
        new_growth_code = f'''            # Applied Clamped2 configuration - tightly controlled growth
            alive_mask = (x[:, 3:4, :, :] > {threshold}).float()
            neighbor_sum = F.conv2d(alive_mask, self.growth_kernel, padding=1)
            growth_factor = torch.clamp(torch.sigmoid(neighbor_sum - 1.0) * 0.35 + 0.75, 0.74, 0.96)
            x = x * growth_factor'''
    elif config_name == "Regulated":
        new_growth_code = f'''            # Applied Regulated configuration - self-regulating
            alive_mask = (x[:, 3:4, :, :] > {threshold}).float()
            neighbor_sum = F.conv2d(alive_mask, self.growth_kernel, padding=1)
            growth_factor = 0.8 + 0.1 / (1 + torch.exp(2 * (neighbor_sum - 1.5)))
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

def create_ultra_precise_plot(results):
    """Create visualization of ultra-precise results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Growth curves with stability
    ax = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, (config_name, result) in enumerate(results.items()):
        steps = list(range(0, len(result["alive_counts"]) * 3, 3))
        color = colors[i]
        
        # Line style based on stability
        stability = result.get("stability_score", 0)
        linestyle = '-' if stability > 0.8 else '--' if stability > 0.5 else ':'
        
        ax.plot(steps, result["alive_counts"], 
               color=color, label=f'{config_name} (s={stability:.2f})', 
               linewidth=2, linestyle=linestyle, marker='o', markersize=3)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Alive Cells')
    ax.set_title('Ultra-Precise Growth Curves (line style = stability)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4200)
    
    # Stability vs Coverage scatter
    ax = axes[0, 1]
    coverages = [results[name]["coverage"] for name in results.keys()]
    stabilities = [results[name].get("stability_score", 0) for name in results.keys()]
    colors_scatter = colors[:len(results)]
    
    scatter = ax.scatter(coverages, stabilities, c=colors_scatter, s=100, alpha=0.7)
    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Stability Score')
    ax.set_title('Stability vs Coverage')
    ax.grid(True, alpha=0.3)
    
    # Add labels
    for i, name in enumerate(results.keys()):
        ax.annotate(name, (coverages[i], stabilities[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Show best 2 final states
    sorted_by_stability = sorted(results.items(), 
                                key=lambda x: x[1].get("stability_score", 0), 
                                reverse=True)
    
    for i, (config_name, result) in enumerate(sorted_by_stability[:2]):
        ax = axes[1, i]
        alpha = result["final_state"][0, 3].detach().numpy()
        im = ax.imshow(alpha, cmap='plasma', vmin=0, vmax=1)
        stability = result.get("stability_score", 0)
        ax.set_title(f'{config_name}\n{result["final_alive"]:.0f} cells (stability: {stability:.3f})')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('ultra_precise_nca_results.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved ultra-precise results as 'ultra_precise_nca_results.png'")

if __name__ == "__main__":
    results, healthy = test_ultra_precise_configs()
    print(f"\n✅ Ultra-precise tuning complete! Found {len(healthy)} stable configurations.") 