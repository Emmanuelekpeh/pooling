import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated import IntegratedNCA, Z_DIM, W_DIM, DEVICE, IMG_SIZE

def test_nca_full_growth():
    """Test if NCA can grow to fill the entire 64x64 space"""
    print("Testing NCA growth to full image coverage...")
    
    # Create NCA model
    nca = IntegratedNCA(channel_n=8, w_dim=W_DIM).to(DEVICE)
    nca.eval()
    
    # Create a random style vector
    w = torch.randn(1, W_DIM).to(DEVICE)
    
    # Get initial seed (5x5 in center)
    seed = nca.get_seed(batch_size=1, size=IMG_SIZE, device=DEVICE)
    
    print(f"Initial seed shape: {seed.shape}")
    print(f"Seed center area (alive cells): {torch.sum(seed[0, 3, :, :] > 0.1).item()} pixels")
    
    # Test different step counts to see growth progression
    step_counts = [32, 64, 96, 128, 160, 200]
    
    with torch.no_grad():
        for steps in step_counts:
            # Start from fresh seed each time
            test_seed = seed.clone()
            
            # Run NCA for specified steps
            result = nca(test_seed, w, steps=steps)
            rgba = nca.to_rgba(result)
            
            # Count alive pixels (alpha > 0.1)
            alive_pixels = torch.sum(rgba[0, 3, :, :] > 0.1).item()
            total_pixels = IMG_SIZE * IMG_SIZE
            coverage_percent = (alive_pixels / total_pixels) * 100
            
            print(f"Steps {steps:3d}: {alive_pixels:4d}/{total_pixels} pixels alive ({coverage_percent:5.1f}% coverage)")
            
            # Check if we've achieved near-full coverage
            if coverage_percent > 90:
                print(f"✓ Near-full coverage achieved at {steps} steps!")
                break
            elif coverage_percent > 75:
                print(f"✓ Good coverage achieved at {steps} steps")
            elif coverage_percent < 10:
                print(f"✗ Poor growth - only {coverage_percent:.1f}% coverage")
    
    # Test with maximum steps to see absolute potential
    print(f"\nTesting with maximum steps (300)...")
    test_seed = seed.clone()
    result = nca(test_seed, w, steps=300)
    rgba = nca.to_rgba(result)
    
    alive_pixels = torch.sum(rgba[0, 3, :, :] > 0.1).item()
    total_pixels = IMG_SIZE * IMG_SIZE
    coverage_percent = (alive_pixels / total_pixels) * 100
    
    print(f"Max steps (300): {alive_pixels}/{total_pixels} pixels alive ({coverage_percent:.1f}% coverage)")
    
    # Visualize the final result
    final_rgb = rgba[0, :3, :, :].detach().cpu()
    final_alpha = rgba[0, 3, :, :].detach().cpu()
    
    # Convert from [-1,1] to [0,1] for display
    final_rgb = (final_rgb + 1.0) / 2.0
    
    # Create RGBA image for visualization
    final_rgba = torch.cat([final_rgb, final_alpha.unsqueeze(0)], dim=0)
    final_rgba = final_rgba.permute(1, 2, 0).numpy()
    
    # Save visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB channels
    ax1.imshow(final_rgb.permute(1, 2, 0).numpy())
    ax1.set_title('RGB Output')
    ax1.axis('off')
    
    # Alpha channel (alive mask)
    ax2.imshow(final_alpha.numpy(), cmap='gray')
    ax2.set_title(f'Alpha Channel\n({coverage_percent:.1f}% coverage)')
    ax2.axis('off')
    
    # Combined RGBA
    ax3.imshow(final_rgba)
    ax3.set_title('Combined RGBA')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('nca_growth_test.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved as 'nca_growth_test.png'")
    
    # Analyze growth pattern
    print(f"\nGrowth Analysis:")
    print(f"- Seed size: 5x5 = 25 pixels")
    print(f"- Final coverage: {alive_pixels} pixels ({coverage_percent:.1f}%)")
    print(f"- Growth factor: {alive_pixels/25:.1f}x")
    
    if coverage_percent > 90:
        print("✓ NCA CAN fill the entire image space!")
        return True
    elif coverage_percent > 50:
        print("⚠ NCA has good growth but may need more steps or parameter tuning")
        return False
    else:
        print("✗ NCA has limited growth - needs significant parameter adjustments")
        return False

if __name__ == "__main__":
    success = test_nca_full_growth()
    
    if not success:
        print("\nSuggestions for improving NCA growth:")
        print("1. Increase NCA_STEPS_MAX beyond 96")
        print("2. Make life_mask threshold even more permissive (< 0.005)")
        print("3. Increase update magnitude (> 0.5)")
        print("4. Increase stochastic update probability (> 0.9)")
        print("5. Use larger neighbor_life kernel (> 7x7)") 