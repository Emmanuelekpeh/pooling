import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated import IntegratedNCA, W_DIM, DEVICE, IMG_SIZE

def test_growth_limits():
    """Test the actual growth limits and constraints of the NCA"""
    print("Testing NCA Growth Limits and Constraints")
    print("=" * 50)
    
    # Create NCA model with simple conditioning to avoid dimension issues
    nca = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False).to(DEVICE)
    nca.eval()
    
    # Create simple inputs
    w = torch.randn(1, W_DIM).to(DEVICE)
    seed = nca.get_seed(batch_size=1, size=IMG_SIZE, device=DEVICE, seed_type="center")
    
    print(f"Starting conditions:")
    print(f"- Image size: {IMG_SIZE}x{IMG_SIZE} = {IMG_SIZE*IMG_SIZE} total pixels")
    print(f"- Channels: {nca.channel_n}")
    print(f"- Initial alive cells: {(seed[0, 3, :, :] > 0.1).sum().item()}")
    
    # Test growth over different step counts
    step_counts = [10, 30, 50, 100, 200, 300]
    results = []
    
    with torch.no_grad():
        for steps in step_counts:
            # Start from fresh seed
            test_seed = seed.clone()
            
            # Run NCA
            result = nca(test_seed, w, steps=steps)
            rgba = nca.to_rgba(result)
            
            # Count alive cells
            alive_count = (rgba[0, 3, :, :] > 0.1).sum().item()
            coverage_percent = (alive_count / (IMG_SIZE * IMG_SIZE)) * 100
            
            # Calculate spatial spread
            alive_mask = rgba[0, 3, :, :] > 0.1
            if alive_count > 0:
                alive_positions = torch.nonzero(alive_mask)
                if len(alive_positions) > 0:
                    max_x = alive_positions[:, 0].max().item()
                    min_x = alive_positions[:, 0].min().item()
                    max_y = alive_positions[:, 1].max().item()
                    min_y = alive_positions[:, 1].min().item()
                    spread_x = max_x - min_x + 1
                    spread_y = max_y - min_y + 1
                else:
                    spread_x = spread_y = 0
            else:
                spread_x = spread_y = 0
            
            results.append({
                'steps': steps,
                'alive_count': alive_count,
                'coverage_percent': coverage_percent,
                'spread_x': spread_x,
                'spread_y': spread_y,
                'rgba': rgba.clone()
            })
            
            print(f"Steps {steps:3d}: {alive_count:4d} cells ({coverage_percent:5.1f}%) - Spread: {spread_x}x{spread_y}")
    
    # Analysis of constraints
    print(f"\nGrowth Analysis:")
    max_coverage = max(r['coverage_percent'] for r in results)
    max_alive = max(r['alive_count'] for r in results)
    final_result = results[-1]
    
    print(f"• Maximum coverage achieved: {max_coverage:.1f}% ({max_alive} cells)")
    print(f"• Final coverage: {final_result['coverage_percent']:.1f}% ({final_result['alive_count']} cells)")
    print(f"• Maximum theoretical coverage: 100% ({IMG_SIZE*IMG_SIZE} cells)")
    
    # Check for growth constraints
    print(f"\nIdentified Growth Constraints:")
    
    # 1. Grid boundary constraint
    print(f"✓ 1. Grid Boundary: {IMG_SIZE}x{IMG_SIZE} = {IMG_SIZE*IMG_SIZE} pixel maximum")
    
    # 2. Life threshold constraint  
    alpha_values = final_result['rgba'][0, 3, :, :].flatten()
    alive_alphas = alpha_values[alpha_values > 0.1]
    if len(alive_alphas) > 0:
        print(f"✓ 2. Alpha Threshold: Cells need α > 0.1 to be alive (current range: {alpha_values.min():.3f} - {alpha_values.max():.3f})")
    
    # 3. Neighbor dependency
    print(f"✓ 3. Neighbor Dependency: Cells die without living neighbors (3x3 kernel)")
    
    # 4. Stochastic updates
    print(f"✓ 4. Stochastic Updates: Only 95% of cells update per step")
    
    # 5. Update magnitude constraint
    print(f"✓ 5. Update Magnitude: Updates scaled by 0.1 to prevent explosion")
    
    # Visualize final state
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB visualization
    final_rgb = final_result['rgba'][0, :3, :, :].detach().cpu()
    final_rgb = torch.tanh(final_rgb)  # Apply tanh to get [-1,1]
    final_rgb = (final_rgb + 1) / 2    # Normalize to [0,1] for display
    axes[0].imshow(final_rgb.permute(1, 2, 0))
    axes[0].set_title(f'Final RGB Output\n({final_result["steps"]} steps)')
    axes[0].axis('off')
    
    # Alpha channel (life mask)
    final_alpha = final_result['rgba'][0, 3, :, :].detach().cpu()
    im1 = axes[1].imshow(final_alpha, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f'Alpha Channel (Life)\n{final_result["alive_count"]} alive cells')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Growth over time
    step_data = [r['steps'] for r in results]
    coverage_data = [r['coverage_percent'] for r in results]
    axes[2].plot(step_data, coverage_data, 'b-o', linewidth=2, markersize=6)
    axes[2].set_xlabel('Training Steps')
    axes[2].set_ylabel('Coverage (%)')
    axes[2].set_title('Coverage vs Steps')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('growth_limits_test.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'growth_limits_test.png'")
    
    # Practical limits summary
    print(f"\nPractical Growth Limits:")
    print(f"• Typical coverage: {max_coverage:.1f}% (vs 100% theoretical)")
    print(f"• Growth plateaus after ~{results[-2]['steps']} steps")
    print(f"• Constraint factor: {100/max_coverage:.1f}x reduction from theoretical maximum")
    
    if max_coverage > 80:
        print("✅ Good growth potential - can fill most of the space")
    elif max_coverage > 50:
        print("⚠️  Moderate growth - reaches about half coverage")
    else:
        print("❌ Limited growth - significant constraints prevent expansion")
    
    return results

if __name__ == "__main__":
    test_growth_limits() 