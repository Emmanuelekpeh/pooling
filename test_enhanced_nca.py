import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated import IntegratedNCA, IntegratedGenerator, Z_DIM, W_DIM, DEVICE, IMG_SIZE

def test_enhanced_nca():
    """Test if the enhanced NCA with rich conditioning works properly"""
    print("Testing Enhanced NCA with Rich Conditioning...")
    
    # Create models
    generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
    nca_basic = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False).to(DEVICE)
    nca_enhanced = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=True).to(DEVICE)
    
    # Create sample inputs
    batch_size = 1
    noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
    fake_target = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)  # Simulated target image
    
    # Generate style vector
    with torch.no_grad():
        _, w = generator(noise, return_w=True)
    
    print(f"Style vector w shape: {w.shape}")
    print(f"Target image shape: {fake_target.shape}")
    
    # Test basic NCA (original)
    print("\n--- Testing Basic NCA ---")
    seed_basic = nca_basic.get_seed(batch_size=batch_size, size=IMG_SIZE, device=DEVICE)
    with torch.no_grad():
        output_basic = nca_basic(seed_basic, w, steps=32)
        print(f"Basic NCA output shape: {output_basic.shape}")
        print(f"Basic NCA alive pixels: {torch.sum(output_basic[0, 3, :, :] > 0.1).item()}")
    
    # Test enhanced NCA (with rich conditioning)
    print("\n--- Testing Enhanced NCA ---")
    seed_enhanced = nca_enhanced.get_seed(batch_size=batch_size, size=IMG_SIZE, device=DEVICE)
    with torch.no_grad():
        output_enhanced = nca_enhanced(seed_enhanced, w, steps=32, target_img=fake_target)
        print(f"Enhanced NCA output shape: {output_enhanced.shape}")
        print(f"Enhanced NCA alive pixels: {torch.sum(output_enhanced[0, 3, :, :] > 0.1).item()}")
    
    # Compare parameter counts
    basic_params = sum(p.numel() for p in nca_basic.parameters())
    enhanced_params = sum(p.numel() for p in nca_enhanced.parameters())
    
    print(f"\n--- Parameter Comparison ---")
    print(f"Basic NCA parameters: {basic_params:,}")
    print(f"Enhanced NCA parameters: {enhanced_params:,}")
    print(f"Additional parameters: {enhanced_params - basic_params:,}")
    
    # Test the conditioning dimensions
    print(f"\n--- Conditioning Analysis ---")
    print(f"W_DIM (style vector): {W_DIM}")
    
    if nca_enhanced.use_rich_conditioning:
        # Test the target encoder
        with torch.no_grad():
            target_features = nca_enhanced.target_encoder(fake_target)
            print(f"Target features shape: {target_features.shape}")
            combined_conditioning = torch.cat([w, target_features], dim=1)
            print(f"Combined conditioning shape: {combined_conditioning.shape}")
            print(f"Total conditioning dimensions: {combined_conditioning.shape[1]}")
    
    print(f"NCA input dimensions: {8 * 3} (perception) + {combined_conditioning.shape[1]} (conditioning) = {8 * 3 + combined_conditioning.shape[1]}")
    
    print("\n✅ Enhanced NCA test completed successfully!")

def test_enhanced_runoff():
    """Test the enhanced runoff mechanism for better spatial exploration"""
    print("🚀 Testing Enhanced NCA Runoff for Spatial Exploration")
    print("=" * 60)
    
    # Create NCA model
    nca = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False).to(DEVICE)
    nca.eval()
    
    # Simple conditioning
    w = torch.randn(1, W_DIM).to(DEVICE)
    
    # Test different seed configurations to see spatial spread
    seed_types = ["center", "distributed", "corners", "line"]
    
    fig, axes = plt.subplots(len(seed_types), 5, figsize=(20, 16))
    
    for row, seed_type in enumerate(seed_types):
        print(f"\n{seed_type.upper()} SEEDING - Enhanced Runoff Test:")
        print("-" * 40)
        
        seed = nca.get_seed(batch_size=1, size=IMG_SIZE, device=DEVICE, seed_type=seed_type)
        
        # Test growth at different steps to see spatial progression
        test_steps = [0, 25, 50, 75, 100]
        
        current_x = seed.clone()
        
        for col, steps in enumerate(test_steps):
            if steps == 0:
                # Show initial seed
                rgba = nca.to_rgba(current_x)
            else:
                # Grow by incremental steps
                step_increment = steps - (test_steps[col-1] if col > 0 else 0)
                current_x = nca(current_x, w, steps=step_increment)
                rgba = nca.to_rgba(current_x)
            
            # Count alive cells and measure spatial spread
            alive_mask = (rgba[:, 3:4, :, :] > 0.1).float()
            alive_count = alive_mask.sum().item()
            
            # Measure spatial spread using center of mass dispersion
            if alive_count > 0:
                # Get coordinates of alive cells
                alive_coords = torch.nonzero(alive_mask[0, 0, :, :], as_tuple=False).float()
                if len(alive_coords) > 1:
                    # Calculate standard deviation of positions (spread measure)
                    center_x = alive_coords[:, 0].mean()
                    center_y = alive_coords[:, 1].mean()
                    spread_x = ((alive_coords[:, 0] - center_x) ** 2).mean().sqrt()
                    spread_y = ((alive_coords[:, 1] - center_y) ** 2).mean().sqrt()
                    spatial_spread = (spread_x + spread_y) / 2
                else:
                    spatial_spread = 0.0
            else:
                spatial_spread = 0.0
            
            print(f"  Step {steps:3d}: {alive_count:4.0f} cells, Spread: {spatial_spread:5.2f}, Coverage: {alive_count/4096*100:4.1f}%")
            
            # Visualize
            img_to_show = rgba[0, :3, :, :].detach().cpu()
            img_to_show = torch.clamp((img_to_show + 1) / 2, 0, 1)  # Normalize to [0,1]
            img_to_show = img_to_show.permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(img_to_show)
            axes[row, col].set_title(f"{seed_type}\nStep {steps}\n{alive_count:.0f} cells\nSpread: {spatial_spread:.1f}")
            axes[row, col].axis('off')
            
            # Add grid overlay to show spatial structure
            for i in range(0, IMG_SIZE, 8):
                axes[row, col].axhline(y=i, color='white', alpha=0.3, linewidth=0.5)
                axes[row, col].axvline(x=i, color='white', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.suptitle("Enhanced NCA Runoff - Spatial Exploration Test", fontsize=16, y=0.98)
    plt.savefig('enhanced_nca_runoff_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("✅ Enhanced runoff test completed!")
    print("Look for:")
    print("  - Cells spreading BEYOND initial clusters")
    print("  - Higher spatial spread values over time")
    print("  - Edge expansion and long-range connections")
    print("  - Less clustering, more exploration")

def test_edge_expansion_mechanism():
    """Specifically test the edge expansion mask functionality"""
    print("\n🎯 Testing Edge Expansion Mechanism")
    print("=" * 40)
    
    nca = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False).to(DEVICE)
    
    # Create a small test pattern to see edge detection
    test_grid = torch.zeros(1, 8, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Create a small cluster in the center
    center = IMG_SIZE // 2
    test_grid[:, 3, center-2:center+3, center-2:center+3] = 1.0  # 5x5 alive region
    
    # Test edge expansion mask
    alive_mask = test_grid[:, 3:4, :, :]
    edge_mask = nca._get_edge_expansion_mask(alive_mask)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original alive mask
    axes[0].imshow(alive_mask[0, 0, :, :].detach().cpu().numpy(), cmap='Blues')
    axes[0].set_title('Original Alive Cells')
    axes[0].axis('off')
    
    # Edge expansion mask
    axes[1].imshow(edge_mask[0, 0, :, :].detach().cpu().numpy(), cmap='Reds')
    axes[1].set_title('Edge Expansion Opportunities')
    axes[1].axis('off')
    
    # Combined
    combined = alive_mask[0, 0, :, :] * 0.5 + edge_mask[0, 0, :, :]
    axes[2].imshow(combined.detach().cpu().numpy(), cmap='RdYlBu')
    axes[2].set_title('Combined: Blue=Alive, Red=Expansion')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_expansion_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Count expansion opportunities
    expansion_count = (edge_mask > 0.1).sum().item()
    alive_count = (alive_mask > 0.1).sum().item()
    
    print(f"Alive cells: {alive_count}")
    print(f"Expansion opportunities: {expansion_count}")
    print(f"Expansion ratio: {expansion_count/alive_count:.2f}")
    print("✅ Edge expansion mechanism tested!")

if __name__ == "__main__":
    test_enhanced_nca()
    test_enhanced_runoff()
    test_edge_expansion_mechanism()
    print("\n🎉 All enhanced runoff tests completed!") 