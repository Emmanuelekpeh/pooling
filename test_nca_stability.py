import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from train_integrated import IntegratedNCA, Z_DIM, W_DIM, DEVICE, IMG_SIZE

def test_nca_stability():
    """Test NCA for potential stability issues based on GAN training best practices"""
    print("Testing NCA Stability and Best Practices...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    channel_n = 8
    w_dim = W_DIM
    
    # Create NCA model
    nca = IntegratedNCA(channel_n=channel_n, w_dim=w_dim, use_rich_conditioning=True).to(DEVICE)
    
    # Test 1: Weight initialization check
    print("\n1. Testing weight initialization...")
    
    weight_stats = {}
    for name, param in nca.named_parameters():
        if param.requires_grad:
            weight_stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            }
    
    print("  Weight statistics:")
    for name, stats in weight_stats.items():
        if 'weight' in name:  # Focus on weight layers
            print(f"    {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")
            
            # Check for potential issues based on GAN stability research
            if abs(stats['mean']) > 0.1:
                print(f"      ⚠️  Large weight mean detected")
            if stats['std'] > 2.0:
                print(f"      ⚠️  Large weight variance detected")
            if stats['std'] < 0.01:
                print(f"      ⚠️  Very small weight variance detected")
    
    # Test 2: Gradient magnitude analysis
    print("\n2. Testing gradient magnitude and flow...")
    
    seed = nca.get_seed(batch_size=batch_size, size=IMG_SIZE, device=DEVICE)
    w = torch.randn(batch_size, w_dim).to(DEVICE)
    target_img = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Enable gradients
    seed.requires_grad_(True)
    w.requires_grad_(True)
    target_img.requires_grad_(True)
    
    # Forward pass
    output = nca(seed, w, steps=10, target_img=target_img)
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients
    gradient_stats = {}
    for name, param in nca.named_parameters():
        if param.grad is not None:
            grad = param.grad
            gradient_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item()
            }
    
    print("  Gradient statistics:")
    for name, stats in gradient_stats.items():
        print(f"    {name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, max={stats['max']:.6f}")
        
        # Check for gradient issues
        if stats['max'] > 10.0:
            print(f"      ⚠️  Large gradient detected - potential exploding gradients")
        if stats['max'] < 1e-6:
            print(f"      ⚠️  Very small gradient detected - potential vanishing gradients")
    
    # Test 3: Output range and stability
    print("\n3. Testing output range and stability...")
    
    # Test multiple forward passes with same inputs
    outputs = []
    for i in range(5):
        torch.manual_seed(42 + i)  # Different random seeds for stochastic updates
        out = nca(seed.clone().detach(), w.clone().detach(), steps=20, target_img=target_img.clone().detach())
        outputs.append(out)
    
    # Analyze output stability
    output_tensor = torch.stack(outputs)
    output_mean = output_tensor.mean(dim=0)
    output_std = output_tensor.std(dim=0)
    
    print(f"  Output range: [{output_tensor.min().item():.3f}, {output_tensor.max().item():.3f}]")
    print(f"  Output stability (std across runs): {output_std.mean().item():.6f}")
    
    # Check for instability
    if output_std.mean().item() > 1.0:
        print(f"    ⚠️  High output variance detected - potential instability")
    
    # Test 4: Life dynamics stability
    print("\n4. Testing life dynamics...")
    
    # Test growth over longer periods
    seed_test = nca.get_seed(batch_size=1, size=IMG_SIZE, device=DEVICE)
    alive_counts = []
    
    for steps in [10, 20, 30, 50, 80, 100]:
        output = nca(seed_test.clone(), w[:1], steps=steps, target_img=target_img[:1])
        alive_count = (output[:, 3, :, :] > 0.1).sum().item()
        alive_counts.append(alive_count)
    
    print("  Life dynamics over time:")
    for i, (steps, count) in enumerate(zip([10, 20, 30, 50, 80, 100], alive_counts)):
        print(f"    Steps {steps}: {count} alive cells")
        
        # Check for potential issues
        if i > 0:
            if count == 0:
                print(f"      ⚠️  Complete death detected at step {steps}")
            if count > IMG_SIZE * IMG_SIZE * 0.8:
                print(f"      ⚠️  Potential runaway growth detected at step {steps}")
    
    # Test 5: Conditioning pathway balance
    print("\n5. Testing conditioning pathway balance...")
    
    # Get pathway outputs
    target_features = nca.target_encoder(target_img[:1])
    w_semantic = nca.w_projection(w[:1])
    content_updates = nca.content_pathway(target_features)
    style_updates = nca.style_pathway(w_semantic)
    
    print(f"  Target features magnitude: {target_features.abs().mean().item():.6f}")
    print(f"  W semantic magnitude: {w_semantic.abs().mean().item():.6f}")
    print(f"  Content updates magnitude: {content_updates.abs().mean().item():.6f}")
    print(f"  Style updates magnitude: {style_updates.abs().mean().item():.6f}")
    
    # Check for pathway dominance issues
    content_mag = content_updates.abs().mean().item()
    style_mag = style_updates.abs().mean().item()
    
    if content_mag > style_mag * 10:
        print(f"    ⚠️  Content pathway strongly dominates style pathway")
    elif style_mag > content_mag * 10:
        print(f"    ⚠️  Style pathway strongly dominates content pathway")
    else:
        print(f"    ✓ Pathways reasonably balanced")
    
    # Test 6: Activation function analysis
    print("\n6. Testing activation patterns...")
    
    # Analyze activations in different layers
    with torch.no_grad():
        # Get intermediate activations
        target_encoded = nca.target_encoder(target_img[:1])
        
        # Check for saturation
        relu_saturated = (target_encoded == 0).float().mean().item()
        print(f"  ReLU saturation in target encoder: {relu_saturated*100:.1f}%")
        
        if relu_saturated > 0.5:
            print(f"    ⚠️  High ReLU saturation detected - potential dead neurons")
    
    # Test 7: Memory usage pattern
    print("\n7. Testing memory usage patterns...")
    
    # Test memory efficiency with longer sequences
    try:
        # Gradually increase steps to test memory scaling
        for steps in [10, 25, 50, 100]:
            output = nca(seed.clone(), w, steps=steps, target_img=target_img)
            print(f"  ✓ {steps} steps completed successfully")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    ⚠️  Memory issue detected: {e}")
        else:
            raise
    
    print("\n" + "=" * 60)
    print("🔍 NCA STABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Summary recommendations based on findings
    print("\nStability Recommendations:")
    print("• Check weight initialization follows best practices")
    print("• Monitor gradient magnitudes during training")
    print("• Ensure life dynamics remain stable over time")
    print("• Balance conditioning pathway influences")
    print("• Watch for activation saturation issues")
    print("• Monitor memory usage in longer sequences")

if __name__ == "__main__":
    test_nca_stability() 