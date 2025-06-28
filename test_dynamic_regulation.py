import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated_fast import IntegratedNCA

def test_dynamic_regulation():
    """Test the new dynamic regulation system"""
    device = torch.device("cpu")
    
    print("🎛️  Testing Dynamic Regulation System...")
    print("=" * 60)
    
    # Create NCA instance with the new dynamic regulation
    nca = IntegratedNCA(channel_n=8, w_dim=128, hidden_n=64)
    nca.eval()
    
    # Test multiple scenarios
    scenarios = {
        "Single_Seed": {"seed_type": "center", "description": "Single center seed"},
        "Multi_Seed": {"seed_type": "distributed", "description": "Multiple distributed seeds"},
        "Sparse_Start": {"seed_type": "distributed", "description": "Sparse starting condition"},
    }
    
    results = {}
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n🧪 Testing {scenario_name}: {scenario['description']}")
        
        # Create test seed
        batch_size = 1
        img_size = 64
        seed = nca.get_seed(batch_size, img_size, device, seed_type=scenario["seed_type"])
        
        # Create dummy w vector
        w = torch.randn(batch_size, 128, device=device)
        
        # Track detailed evolution
        x = seed.clone()
        alive_counts = []
        alive_ratios = []
        regulation_states = []
        
        print(f"  Initial alive cells: {(x[0, 3] > 0.04).sum().item()}")
        
        # Run extended simulation
        for step_group in range(15):  # More steps to see regulation in action
            if step_group > 0:
                # Run NCA steps
                x = nca(x, w, steps=3)
            
            # Analyze current state
            alive_mask = (x[0, 3] > 0.04).float()
            alive_count = alive_mask.sum().item()
            alive_ratio = alive_count / (64 * 64)
            
            alive_counts.append(alive_count)
            alive_ratios.append(alive_ratio)
            
            # Determine regulation state
            if alive_ratio < 0.02:
                reg_state = "BOOST"
            elif alive_ratio > 0.4:
                reg_state = "SUPPRESS"
            else:
                reg_state = "BALANCE"
            
            regulation_states.append(reg_state)
            
            print(f"    Step {step_group*3:2d}: {alive_count:4.0f} cells ({alive_ratio*100:5.1f}%) - {reg_state}")
            
            # Early termination if dead
            if alive_count == 0:
                print(f"    💀 Population died at step {step_group*3}")
                break
        
        final_alive = alive_counts[-1]
        final_ratio = alive_ratios[-1]
        
        # Analyze regulation effectiveness
        regulation_changes = len(set(regulation_states))
        stability_score = 1.0 - (np.std(alive_counts[-5:]) / (np.mean(alive_counts[-5:]) + 1e-6))
        stability_score = max(0, min(1, stability_score))
        
        # Determine final status
        if final_alive == 0:
            status = "DEAD"
        elif final_alive == 4096:
            status = "OVERGROWN"
        elif final_alive < 50:
            status = "DYING"
        elif final_alive > 2500:
            status = "TOO_DENSE"
        elif 50 <= final_alive <= 2500:
            if stability_score > 0.7:
                status = "REGULATED"
            else:
                status = "HEALTHY"
        else:
            status = "UNKNOWN"
        
        results[scenario_name] = {
            "alive_counts": alive_counts,
            "alive_ratios": alive_ratios,
            "regulation_states": regulation_states,
            "final_alive": final_alive,
            "final_ratio": final_ratio,
            "status": status,
            "stability_score": stability_score,
            "regulation_changes": regulation_changes,
            "final_state": x.clone()
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DYNAMIC REGULATION RESULTS:")
    print("=" * 60)
    
    success_count = 0
    for scenario_name, result in results.items():
        status_emoji = {
            "REGULATED": "🎯",
            "HEALTHY": "✅",
            "OVERGROWN": "⚠️ ",
            "DEAD": "❌",
            "DYING": "⚠️ ",
            "TOO_DENSE": "⚠️ "
        }.get(result["status"], "❓")
        
        if result["status"] in ["REGULATED", "HEALTHY"]:
            success_count += 1
        
        print(f"{status_emoji} {scenario_name:12s}: {result['final_alive']:4.0f} cells "
              f"({result['final_ratio']*100:5.1f}%) - {result['status']} "
              f"(stability: {result['stability_score']:.2f}, reg_changes: {result['regulation_changes']})")
    
    if success_count > 0:
        print(f"\n🎉 SUCCESS! {success_count}/{len(scenarios)} scenarios achieved stable regulation!")
    else:
        print(f"\n😞 No scenarios achieved stable regulation")
    
    # Create regulation visualization
    create_regulation_plot(results)
    
    return results

def create_regulation_plot(results):
    """Create visualization of regulation system performance"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Population evolution
    ax = axes[0, 0]
    for i, (scenario_name, result) in enumerate(results.items()):
        steps = list(range(0, len(result["alive_counts"]) * 3, 3))
        color = colors[i % len(colors)]
        
        ax.plot(steps, result["alive_counts"], 
               color=color, label=scenario_name, 
               linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Alive Cells')
    ax.set_title('Population Evolution with Dynamic Regulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4200)
    
    # Population ratio evolution
    ax = axes[0, 1]
    for i, (scenario_name, result) in enumerate(results.items()):
        steps = list(range(0, len(result["alive_ratios"]) * 3, 3))
        color = colors[i % len(colors)]
        
        ax.plot(steps, [r*100 for r in result["alive_ratios"]], 
               color=color, label=scenario_name, 
               linewidth=2, marker='s', markersize=3)
    
    # Add regulation zones
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Boost threshold')
    ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Suppress threshold')
    ax.fill_between([0, 50], 2, 40, alpha=0.1, color='green', label='Balanced zone')
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Population Ratio (%)')
    ax.set_title('Population Ratio with Regulation Zones')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Show best 2 final states
    regulated_results = [(name, res) for name, res in results.items() 
                        if res["status"] in ["REGULATED", "HEALTHY"]]
    if not regulated_results:
        # Show best 2 by stability
        regulated_results = sorted(results.items(), 
                                  key=lambda x: x[1]["stability_score"], 
                                  reverse=True)[:2]
    
    for i, (scenario_name, result) in enumerate(regulated_results[:2]):
        ax = axes[1, i]
        alpha = result["final_state"][0, 3].detach().numpy()
        im = ax.imshow(alpha, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'{scenario_name}\n{result["final_alive"]:.0f} cells ({result["status"]})')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('dynamic_regulation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved regulation results as 'dynamic_regulation_results.png'")

if __name__ == "__main__":
    results = test_dynamic_regulation()
    print(f"\n✅ Dynamic regulation test complete!") 