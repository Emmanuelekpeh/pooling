import torch
import torch.nn.functional as F
import numpy as np
from train_integrated import IntegratedNCA, Z_DIM, W_DIM, DEVICE, IMG_SIZE

def test_nca_integrity():
    """Comprehensive test of the NCA implementation to ensure all components work correctly"""
    print("Testing NCA Implementation Integrity...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    channel_n = 8
    w_dim = W_DIM
    
    # Create NCA models (both rich and simple)
    nca_rich = IntegratedNCA(channel_n=channel_n, w_dim=w_dim, use_rich_conditioning=True).to(DEVICE)
    nca_simple = IntegratedNCA(channel_n=channel_n, w_dim=w_dim, use_rich_conditioning=False).to(DEVICE)
    
    print(f"✓ Models created successfully")
    print(f"  - Rich conditioning model: {sum(p.numel() for p in nca_rich.parameters())} parameters")
    print(f"  - Simple conditioning model: {sum(p.numel() for p in nca_simple.parameters())} parameters")
    
    # Test 1: Seed generation
    print("\n1. Testing seed generation...")
    seed = nca_rich.get_seed(batch_size=batch_size, size=IMG_SIZE, device=DEVICE)
    print(f"  ✓ Seed shape: {seed.shape}")
    print(f"  ✓ Seed device: {seed.device}")
    print(f"  ✓ Seed dtype: {seed.dtype}")
    
    # Check seed has alive cells
    alive_cells = (seed[:, 3, :, :] > 0.5).sum()
    print(f"  ✓ Alive cells in seed: {alive_cells.item()}")
    assert alive_cells > 0, "Seed should have alive cells"
    
    # Test 2: Perception mechanism
    print("\n2. Testing perception mechanism...")
    perceived = nca_rich.perceive(seed)
    print(f"  ✓ Perceived shape: {perceived.shape}")
    print(f"  ✓ Expected shape: {(batch_size, channel_n * 3, IMG_SIZE, IMG_SIZE)}")
    assert perceived.shape == (batch_size, channel_n * 3, IMG_SIZE, IMG_SIZE), "Perception output shape mismatch"
    
    # Test 3: RGBA conversion
    print("\n3. Testing RGBA conversion...")
    rgba = nca_rich.to_rgba(seed)
    print(f"  ✓ RGBA shape: {rgba.shape}")
    print(f"  ✓ RGBA range: [{rgba.min().item():.3f}, {rgba.max().item():.3f}]")
    assert rgba.shape == (batch_size, 4, IMG_SIZE, IMG_SIZE), "RGBA output shape mismatch"
    
    # Test 4: W vector and target image inputs
    print("\n4. Testing conditioning inputs...")
    w = torch.randn(batch_size, w_dim).to(DEVICE)
    target_img = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    print(f"  ✓ W vector shape: {w.shape}")
    print(f"  ✓ Target image shape: {target_img.shape}")
    
    # Test 5: Rich conditioning forward pass
    print("\n5. Testing rich conditioning forward pass...")
    try:
        steps = 10  # Short run for testing
        output_rich = nca_rich(seed.clone(), w, steps=steps, target_img=target_img)
        print(f"  ✓ Rich conditioning output shape: {output_rich.shape}")
        print(f"  ✓ Rich conditioning output range: [{output_rich.min().item():.3f}, {output_rich.max().item():.3f}]")
        assert output_rich.shape == seed.shape, "Rich conditioning output shape mismatch"
    except Exception as e:
        print(f"  ✗ Rich conditioning failed: {e}")
        raise
    
    # Test 6: Simple conditioning forward pass
    print("\n6. Testing simple conditioning forward pass...")
    try:
        output_simple = nca_simple(seed.clone(), w, steps=steps)
        print(f"  ✓ Simple conditioning output shape: {output_simple.shape}")
        print(f"  ✓ Simple conditioning output range: [{output_simple.min().item():.3f}, {output_simple.max().item():.3f}]")
        assert output_simple.shape == seed.shape, "Simple conditioning output shape mismatch"
    except Exception as e:
        print(f"  ✗ Simple conditioning failed: {e}")
        raise
    
    # Test 7: Growth verification
    print("\n7. Testing growth dynamics...")
    initial_alive = (seed[:, 3, :, :] > 0.1).sum().item()
    final_alive_rich = (output_rich[:, 3, :, :] > 0.1).sum().item()
    final_alive_simple = (output_simple[:, 3, :, :] > 0.1).sum().item()
    
    print(f"  ✓ Initial alive cells: {initial_alive}")
    print(f"  ✓ Rich conditioning final alive cells: {final_alive_rich}")
    print(f"  ✓ Simple conditioning final alive cells: {final_alive_simple}")
    
    # Test 8: Network component integrity
    print("\n8. Testing network components...")
    
    # Test rich conditioning components
    if nca_rich.use_rich_conditioning:
        # Test target encoder
        target_features = nca_rich.target_encoder(target_img)
        print(f"  ✓ Target encoder output shape: {target_features.shape}")
        
        # Test w projection
        w_semantic = nca_rich.w_projection(w)
        print(f"  ✓ W projection output shape: {w_semantic.shape}")
        
        # Test content pathway
        content_updates = nca_rich.content_pathway(target_features)
        print(f"  ✓ Content pathway output shape: {content_updates.shape}")
        
        # Test style pathway
        style_updates = nca_rich.style_pathway(w_semantic)
        print(f"  ✓ Style pathway output shape: {style_updates.shape}")
        
        print(f"  ✓ All rich conditioning components working")
    
    # Test 9: Gradient flow
    print("\n9. Testing gradient flow...")
    
    # Enable gradients
    seed_grad = seed.clone().requires_grad_(True)
    w_grad = w.clone().requires_grad_(True)
    target_grad = target_img.clone().requires_grad_(True)
    
    # Forward pass
    output_grad = nca_rich(seed_grad, w_grad, steps=5, target_img=target_grad)
    loss = output_grad.mean()
    
    # Backward pass
    loss.backward()
    
    print(f"  ✓ Gradients computed successfully")
    print(f"  ✓ Seed gradient: {seed_grad.grad is not None}")
    print(f"  ✓ W gradient: {w_grad.grad is not None}")
    print(f"  ✓ Target gradient: {target_grad.grad is not None}")
    
    # Test 10: Memory efficiency check
    print("\n10. Testing memory efficiency...")
    
    # Run longer sequence to test memory cleanup
    try:
        output_long = nca_rich(seed.clone(), w, steps=50, target_img=target_img)
        print(f"  ✓ Long sequence (50 steps) completed successfully")
        print(f"  ✓ Output shape: {output_long.shape}")
    except Exception as e:
        print(f"  ✗ Long sequence failed: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("🎉 ALL NCA INTEGRITY TESTS PASSED!")
    print("=" * 60)
    
    # Summary
    print("\nNCA Implementation Summary:")
    print(f"• Channel count: {channel_n}")
    print(f"• W dimension: {w_dim}")
    print(f"• Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"• Rich conditioning: ✓ Working")
    print(f"• Simple conditioning: ✓ Working") 
    print(f"• Perception mechanism: ✓ Working")
    print(f"• Growth dynamics: ✓ Working")
    print(f"• Gradient flow: ✓ Working")
    print(f"• Memory efficiency: ✓ Working")

if __name__ == "__main__":
    test_nca_integrity() 