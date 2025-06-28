import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated import IntegratedNCA, IntegratedGenerator, Z_DIM, W_DIM, DEVICE, IMG_SIZE

def test_stylegan_conditioning():
    """Test the new StyleGAN-inspired conditioning with proper weight distribution"""
    print("Testing StyleGAN-Inspired NCA Conditioning...")
    
    # Create models
    generator = IntegratedGenerator(Z_DIM, W_DIM).to(DEVICE)
    nca_rich = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=True).to(DEVICE)
    nca_simple = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False).to(DEVICE)
    
    # Create sample inputs
    batch_size = 1
    noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
    fake_target = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Generate w vector and NCA seeds
    with torch.no_grad():
        fake_img, w = generator(noise, return_w=True)
        seed = nca_rich.get_seed(batch_size=batch_size, size=IMG_SIZE, device=DEVICE)
        
        print(f"Input shapes:")
        print(f"  W vector: {w.shape}")
        print(f"  Target image: {fake_target.shape}")
        print(f"  Seed: {seed.shape}")
        
        # Test rich conditioning
        print(f"\n=== RICH CONDITIONING ANALYSIS ===")
        
        # Check w projection (should reduce w to ~30% influence)
        w_semantic = nca_rich.w_projection(w)
        print(f"W vector reduced: {w.shape} -> {w_semantic.shape}")
        print(f"W influence reduction: {w_semantic.shape[1] / w.shape[1] * 100:.1f}% of original")
        
        # Check target encoding (should be ~70% influence)  
        target_features = nca_rich.target_encoder(fake_target)
        print(f"Target features: {target_features.shape}")
        
        # Check conditioning ratio
        total_conditioning = w_semantic.shape[1] + target_features.shape[1]
        w_percentage = w_semantic.shape[1] / total_conditioning * 100
        target_percentage = target_features.shape[1] / total_conditioning * 100
        
        print(f"\nConditioning Distribution:")
        print(f"  W vector: {w_semantic.shape[1]} dims ({w_percentage:.1f}%)")
        print(f"  Target features: {target_features.shape[1]} dims ({target_percentage:.1f}%)")
        print(f"  Total conditioning: {total_conditioning} dims")
        
        # Test pathway outputs
        content_updates = nca_rich.content_pathway(target_features)
        style_updates = nca_rich.style_pathway(w_semantic)
        print(f"\nPathway outputs:")
        print(f"  Content pathway: {content_updates.shape} (target-driven)")
        print(f"  Style pathway: {style_updates.shape} (w-driven)")
        
        # Test full forward pass
        print(f"\n=== FORWARD PASS COMPARISON ===")
        
        # Rich conditioning
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        output_rich = nca_rich(seed.clone(), w, steps=32, target_img=fake_target)
        final_rich = nca_rich.to_rgba(output_rich)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            rich_time = start_time.elapsed_time(end_time)
            print(f"Rich conditioning time: {rich_time:.2f}ms")
        
        # Simple conditioning
        if start_time:
            start_time.record()
            
        output_simple = nca_simple(seed.clone(), w, steps=32)
        final_simple = nca_simple.to_rgba(output_simple)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            simple_time = start_time.elapsed_time(end_time)
            print(f"Simple conditioning time: {simple_time:.2f}ms")
            print(f"Overhead: {((rich_time - simple_time) / simple_time * 100):.1f}%")
        
        # Analyze outputs
        print(f"\nOutput Analysis:")
        print(f"  Rich output range: [{final_rich.min():.3f}, {final_rich.max():.3f}]")
        print(f"  Simple output range: [{final_simple.min():.3f}, {final_simple.max():.3f}]")
        
        # Check alive cells
        rich_alive = (final_rich[0, 3, :, :] > 0.1).sum().item()
        simple_alive = (final_simple[0, 3, :, :] > 0.1).sum().item()
        total_pixels = IMG_SIZE * IMG_SIZE
        
        print(f"  Rich alive cells: {rich_alive}/{total_pixels} ({rich_alive/total_pixels*100:.1f}%)")
        print(f"  Simple alive cells: {simple_alive}/{total_pixels} ({simple_alive/total_pixels*100:.1f}%)")
        
        # Calculate difference
        diff = F.mse_loss(final_rich, final_simple)
        print(f"  MSE difference: {diff.item():.6f}")
        
        print(f"\n=== MEMORY ANALYSIS ===")
        # Count parameters
        rich_params = sum(p.numel() for p in nca_rich.parameters())
        simple_params = sum(p.numel() for p in nca_simple.parameters())
        
        print(f"Rich conditioning parameters: {rich_params:,}")
        print(f"Simple conditioning parameters: {simple_params:,}")
        print(f"Parameter increase: {((rich_params - simple_params) / simple_params * 100):.1f}%")
        
        return {
            'w_percentage': w_percentage,
            'target_percentage': target_percentage,
            'rich_alive': rich_alive,
            'simple_alive': simple_alive,
            'mse_difference': diff.item(),
            'param_increase': (rich_params - simple_params) / simple_params * 100
        }

if __name__ == "__main__":
    results = test_stylegan_conditioning()
    
    print(f"\n=== SUMMARY ===")
    print(f"✅ W vector influence: {results['w_percentage']:.1f}% (target: ~30%)")
    print(f"✅ Target influence: {results['target_percentage']:.1f}% (target: ~70%)")
    print(f"✅ Parameter overhead: +{results['param_increase']:.1f}%")
    print(f"✅ Output difference: {results['mse_difference']:.6f}")
    
    if results['w_percentage'] < 40 and results['target_percentage'] > 60:
        print("🎯 Conditioning distribution is within target range!")
    else:
        print("⚠️  Conditioning distribution needs adjustment") 