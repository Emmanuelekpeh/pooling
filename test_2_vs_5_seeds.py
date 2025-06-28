import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from train_integrated import IntegratedNCA, W_DIM, DEVICE, IMG_SIZE

def test_seed_comparison():
    """Compare 2-seed vs 5-seed distributed patterns"""
    print("🔬 Testing 2-Seed vs 5-Seed Distributed Patterns")
    print("=" * 60)
    
    # Create NCA model
    nca = IntegratedNCA(channel_n=8, w_dim=W_DIM, use_rich_conditioning=False).to(DEVICE)
    nca.eval()
    
    # Simple conditioning
    w = torch.randn(1, W_DIM).to(DEVICE)
    
    # Test both patterns
    patterns = {
        "2-seed": {"num_seeds": 2, "positions": None},
        "5-seed": {"num_seeds": 5, "positions": None}
    }
    
    # Manually create positions for comparison
    size = IMG_SIZE
    
    # 2-seed diagonal pattern
    offset_2 = size // 3
    positions_2 = [
        (offset_2, offset_2),                      # Top-left area
        (size - offset_2 - 1, size - offset_2 - 1), # Bottom-right area
    ]
    
    # 5-seed quincunx pattern (old)
    offset_5 = size // 4
    positions_5 = [
        (offset_5, offset_5),                      # Top-left quadrant
        (size - offset_5 - 1, offset_5),          # Top-right quadrant
        (offset_5, size - offset_5 - 1),          # Bottom-left quadrant  
        (size - offset_5 - 1, size - offset_5 - 1), # Bottom-right quadrant
        (size // 2, size // 2)                    # Center
    ]
    
    patterns["2-seed"]["positions"] = positions_2
    patterns["5-seed"]["positions"] = positions_5
    
    results = {}
    
    for pattern_name, pattern_info in patterns.items():
        print(f"\n{pattern_name.upper()} PATTERN:")
        print("-" * 30)
        
        # Create custom seed
        seed = torch.zeros(1, 8, size, size, device=DEVICE)
        for pos_x, pos_y in pattern_info["positions"]:
            # Place seed with radius 1
            for i in range(-1, 2):
                for j in range(-1, 2):
                    x, y = pos_x + i, pos_y + j
                    if 0 <= x < size and 0 <= y < size:
                        seed[:, 0, x, y] = 0.5  # R
                        seed[:, 1, x, y] = 0.3  # G  
                        seed[:, 2, x, y] = 0.7  # B
                        seed[:, 3, x, y] = 1.0  # Alpha (alive)
        
        # Count initial cells
        initial_alive = (seed[:, 3:4, :, :] > 0.1).sum().item()
        initial_coverage = (initial_alive / (size * size)) * 100
        
        print(f"  Initial: {initial_alive} cells ({initial_coverage:.1f}% coverage)")
        
        # Run NCA
        with torch.no_grad():
            x = seed.clone()
            steps = 50
            
            for step in range(0, steps + 1, 10):
                if step > 0:
                    x = nca(x, w, steps=10)
                
                alive_mask = (x[:, 3:4, :, :] > 0.1).float()
                alive_count = alive_mask.sum().item()
                coverage = (alive_count / (size * size)) * 100
                
                print(f"  Step {step:2d}: {alive_count:4.0f} cells ({coverage:5.1f}% coverage)")
        
        # Final analysis
        final_alive = alive_count
        final_coverage = coverage
        growth_rate = (final_alive - initial_alive) / (steps / 10)
        
        results[pattern_name] = {
            'initial_alive': initial_alive,
            'final_alive': final_alive,
            'final_coverage': final_coverage,
            'growth_rate': growth_rate
        }
    
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS:")
    print("=" * 60)
    
    for pattern_name, result in results.items():
        print(f"\n{pattern_name.upper()}:")
        print(f"  Initial cells: {result['initial_alive']}")
        print(f"  Final cells: {result['final_alive']:.0f}")
        print(f"  Final coverage: {result['final_coverage']:.1f}%")
        print(f"  Growth rate: {result['growth_rate']:.1f} cells/step")
        
        # Efficiency calculation
        efficiency = result['final_coverage'] / result['initial_alive']
        print(f"  Efficiency: {efficiency:.1f}% coverage per initial cell")
    
    # Recommendation
    print("\n🎯 RECOMMENDATION:")
    print("-" * 30)
    
    growth_diff = results['2-seed']['growth_rate'] - results['5-seed']['growth_rate']
    coverage_diff = results['2-seed']['final_coverage'] - results['5-seed']['final_coverage']
    
    if abs(coverage_diff) < 5:  # Similar coverage
        print("✅ 2-seed pattern is RECOMMENDED:")
        print("   - Similar final coverage to 5-seed")
        print("   - More controlled and predictable growth")
        print("   - Less computational overhead")
        print("   - Enhanced runoff makes up for fewer seeds")
    else:
        print("⚠️  Consider keeping 5-seed pattern if coverage difference is significant")
    
    print(f"\nCoverage difference: {coverage_diff:+.1f}%")
    print(f"Growth rate difference: {growth_diff:+.1f} cells/step")

if __name__ == "__main__":
    test_seed_comparison() 