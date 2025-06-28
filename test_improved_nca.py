import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated_fast import IntegratedNCA

def test_improved_nca():
    """Test the improved NCA growth parameters"""
    device = torch.device("cpu")
    
    # Initialize NCA with improved parameters
    nca = IntegratedNCA(channel_n=8, w_dim=128, hidden_n=64)
    nca.eval()
    
    # Create test seed
    batch_size = 1
    img_size = 64
    seed = nca.get_seed(batch_size, img_size, device, seed_type="distributed")
    
    print(f"🌱 Improved NCA Seed:")
    print(f"  Alive cells (>0.03): {(seed[0, 3] > 0.03).sum().item()}")
    print(f"  Alpha max: {seed[0, 3].max().item():.4f}")
    print(f"  Alpha mean: {seed[0, 3].mean().item():.4f}")
    print(f"  RGB range: [{seed[0, :3].min().item():.4f}, {seed[0, :3].max().item():.4f}]")
    
    # Create dummy w vector
    w = torch.randn(batch_size, 128, device=device)
    
    # Test growth over steps
    x = seed.clone()
    alive_counts = []
    alpha_means = []
    
    print(f"\n🔬 Testing growth dynamics:")
    for step in range(0, 31, 5):
        if step > 0:
            x = nca(x, w, steps=5)
        
        alive_mask = (x[0, 3] > 0.02).float()  # Use new threshold
        alive_count = alive_mask.sum().item()
        alpha_mean = x[0, 3].mean().item()
        alive_counts.append(alive_count)
        alpha_means.append(alpha_mean)
        
        print(f"  Step {step:2d}: {alive_count:4.0f} alive cells ({alive_count/4096*100:4.1f}%), alpha_mean={alpha_mean:.4f}")
        
        # Check for issues
        if alive_count == 0:
            print("  ❌ NCA DIED!")
            break
        elif alive_count == 4096:
            print("  ⚠️  NCA OVERGROWN!")
            break
    
    # Analysis
    final_alive = alive_counts[-1]
    growth_ratio = final_alive / alive_counts[0] if alive_counts[0] > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"  Initial → Final: {alive_counts[0]:.0f} → {final_alive:.0f} cells")
    print(f"  Growth ratio: {growth_ratio:.2f}x")
    print(f"  Final coverage: {final_alive/4096*100:.1f}%")
    
    if final_alive == 0:
        status = "❌ DEAD"
    elif final_alive == 4096:
        status = "⚠️  OVERGROWN"
    elif final_alive < 50:
        status = "⚠️  DYING"
    elif final_alive > 2000:
        status = "⚠️  TOO_DENSE"
    elif 50 <= final_alive <= 2000:
        status = "✅ HEALTHY"
    else:
        status = "❓ UNKNOWN"
    
    print(f"  Status: {status}")
    
    # Save visualization
    save_test_visualization(x, status, alive_counts)
    
    return status, alive_counts, x

def save_test_visualization(x, status, alive_counts):
    """Save test visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # RGB output
    rgb = x[0, :3].permute(1, 2, 0)
    rgb = torch.tanh(rgb)
    rgb = (rgb + 1) / 2
    rgb = torch.clamp(rgb, 0, 1)
    
    axes[0,0].imshow(rgb.detach().numpy())
    axes[0,0].set_title(f"RGB Output ({status})")
    axes[0,0].axis('off')
    
    # Alpha channel
    alpha = x[0, 3].detach().numpy()
    im1 = axes[0,1].imshow(alpha, cmap='viridis', vmin=0, vmax=1)
    axes[0,1].set_title(f"Alpha Channel")
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1])
    
    # Alive cells (binary)
    alive_binary = (alpha > 0.02).astype(float)
    axes[1,0].imshow(alive_binary, cmap='RdYlBu_r')
    axes[1,0].set_title(f"Alive Cells (>{len(alive_counts[-1:]) and alive_counts[-1] or 0:.0f})")
    axes[1,0].axis('off')
    
    # Growth curve
    steps = list(range(0, len(alive_counts)*5, 5))
    axes[1,1].plot(steps, alive_counts, 'b-o', linewidth=2, markersize=6)
    axes[1,1].set_xlabel('Steps')
    axes[1,1].set_ylabel('Alive Cells')
    axes[1,1].set_title('Growth Curve')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(0, max(4096, max(alive_counts)*1.1))
    
    plt.tight_layout()
    status_clean = status.replace('❌', 'DEAD').replace('⚠️', 'WARN').replace('✅', 'HEALTHY').replace(' ', '_')
    plt.savefig(f'improved_nca_{status_clean.lower()}.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved as 'improved_nca_{status_clean.lower()}.png'")

if __name__ == "__main__":
    print("🧬 Testing improved NCA parameters...")
    status, counts, output = test_improved_nca()
    print(f"\n🎯 Final result: {status}") 