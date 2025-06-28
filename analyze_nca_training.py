import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_training_results():
    """Analyze the results from our stable NCA training attempts"""
    
    print("🔍 NCA Training Analysis")
    print("=" * 50)
    
    # Look for recent results files
    result_files = [f for f in os.listdir('.') if f.startswith('fast_nca_results_') and f.endswith('.json')]
    
    if not result_files:
        print("❌ No training results found")
        return
    
    # Load the most recent results
    latest_file = max(result_files, key=lambda x: os.path.getmtime(x))
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        print(f"📊 Analyzing results from: {latest_file}")
        print(f"⏱️  Total training time: {results['total_time']:.1f}s")
        print(f"🎯 Overall success: {results['success']}")
        print(f"🔴 Final alive ratio: {results['final_alive']:.4f}")
        print(f"📉 Final loss: {results['final_loss']:.4f}")
        
        print("\n📚 Stage-by-Stage Analysis:")
        print("-" * 40)
        
        for i, stage in enumerate(results['stages']):
            status = "✅" if stage['converged'] else "❌"
            print(f"{status} Stage {i+1}: {stage['stage']}")
            print(f"   Target: {stage['target']:.3f} | Achieved: {stage['final_alive']:.3f}")
            print(f"   Loss: {stage['final_loss']:.4f} | Time: {stage['time']:.1f}s")
            print(f"   Gap: {abs(stage['final_alive'] - stage['target']):.3f}")
            print()
        
        # Analysis insights
        print("🧠 Key Insights:")
        print("-" * 20)
        
        alive_ratios = [s['final_alive'] for s in results['stages']]
        targets = [s['target'] for s in results['stages']]
        
        if all(ratio < 0.02 for ratio in alive_ratios):
            print("⚠️  PROBLEM: NCA is essentially DEAD (alive < 0.02)")
            print("   - The stability-first approach may be too conservative")
            print("   - Need to encourage more initial growth")
            print("   - Consider starting with growth-first curriculum")
        
        if max(alive_ratios) - min(alive_ratios) < 0.001:
            print("⚠️  PROBLEM: No learning progression detected")
            print("   - Alive ratios are not changing between stages")
            print("   - Model may be stuck in local minimum")
            print("   - Need stronger growth incentives")
        
        losses = [s['final_loss'] for s in results['stages']]
        if losses[-1] > losses[0]:
            print("⚠️  PROBLEM: Loss is increasing over stages")
            print("   - Model is not learning target patterns")
            print("   - May need different loss function or architecture")
        
        print("\n💡 Recommendations:")
        print("-" * 20)
        print("1. 🌱 Try GROWTH-FIRST curriculum instead of stability-first")
        print("2. 🔧 Increase initial growth incentives (larger stability_factor)")
        print("3. 🎯 Use different loss weighting (more pattern, less stability early)")
        print("4. 🔄 Consider asynchronous updates to break symmetry")
        print("5. 📏 Try different seed patterns (multiple seeds, larger seeds)")
        
        # Compare to original bifurcation problem
        print("\n🔄 Comparison to Original Problem:")
        print("-" * 35)
        print("BEFORE: NCA had bifurcation (0% alive OR 100% alive)")
        print("NOW:    NCA is stable but essentially dead (~0.9% alive)")
        print("NEEDED: Stable intermediate states (10-25% alive)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return None

def suggest_next_approach():
    """Suggest the next training approach based on our findings"""
    
    print("\n🚀 Suggested Next Approach: GROWTH-FIRST Curriculum")
    print("=" * 55)
    
    approach = {
        "name": "Growth-First Stable NCA",
        "philosophy": "Teach growth first, then add stability constraints",
        "stages": [
            {
                "stage": 1,
                "name": "Aggressive Growth", 
                "goal": "Learn to grow from seed to ~20% alive",
                "stability_weight": 0.1,
                "pattern_weight": 0.9,
                "growth_incentive": 2.0
            },
            {
                "stage": 2,
                "name": "Controlled Growth",
                "goal": "Maintain growth but add basic stability",
                "stability_weight": 0.3,
                "pattern_weight": 0.7,
                "growth_incentive": 1.5
            },
            {
                "stage": 3,
                "name": "Stable Dynamics",
                "goal": "Balance growth and stability",
                "stability_weight": 0.5,
                "pattern_weight": 0.5,
                "growth_incentive": 1.0
            },
            {
                "stage": 4,
                "name": "Pattern Formation",
                "goal": "Learn specific target patterns",
                "stability_weight": 0.4,
                "pattern_weight": 0.8,
                "growth_incentive": 0.8
            },
            {
                "stage": 5,
                "name": "Robust Control",
                "goal": "Maintain patterns under perturbation",
                "stability_weight": 0.6,
                "pattern_weight": 0.6,
                "growth_incentive": 0.6
            }
        ]
    }
    
    for stage in approach["stages"]:
        print(f"📚 Stage {stage['stage']}: {stage['name']}")
        print(f"   Goal: {stage['goal']}")
        print(f"   Weights: Stability={stage['stability_weight']}, Pattern={stage['pattern_weight']}")
        print(f"   Growth incentive: {stage['growth_incentive']}x")
        print()
    
    print("🔧 Key Changes from Previous Approach:")
    print("- Start with GROWTH, not stability")
    print("- Higher growth incentives initially")
    print("- Pattern learning comes before strict stability")
    print("- Gradual introduction of stability constraints")
    
    return approach

if __name__ == "__main__":
    results = analyze_training_results()
    approach = suggest_next_approach()
    
    print(f"\n📝 Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 