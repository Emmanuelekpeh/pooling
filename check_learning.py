#!/usr/bin/env python3

import torch
import numpy as np

def analyze_learning():
    """Check if the system is learning from checkpoint data"""
    
    # Load checkpoint
    checkpoint = torch.load('./checkpoints/latest_checkpoint.pt', weights_only=False)
    scores = checkpoint.get('scores', [])
    
    print(f"📊 Analysis of {len(scores)} epochs:")
    print("=" * 60)
    
    if len(scores) < 10:
        print("❌ Not enough data to analyze learning")
        return
    
    # Get different phases
    early = scores[0:10]
    mid = scores[len(scores)//2-5:len(scores)//2+5] if len(scores) > 20 else scores[10:20]
    late = scores[-10:]
    
    print("🔍 LEARNING ANALYSIS:")
    print()
    
    # NCA Adversarial Loss Analysis
    early_nca = np.mean([s.get('nca_adversarial_loss', 0) for s in early])
    mid_nca = np.mean([s.get('nca_adversarial_loss', 0) for s in mid])
    late_nca = np.mean([s.get('nca_adversarial_loss', 0) for s in late])
    
    print("📈 NCA Adversarial Loss:")
    print(f"  Early (1-10): {early_nca:.4f}")
    print(f"  Mid: {mid_nca:.4f}")
    print(f"  Late: {late_nca:.4f}")
    
    nca_trend = "📈 INCREASING" if late_nca > early_nca else "📉 DECREASING"
    nca_change = abs(late_nca - early_nca)
    print(f"  Trend: {nca_trend} (change: {nca_change:.4f})")
    
    # Generator Adversarial Loss Analysis
    early_gen = np.mean([s.get('gen_adversarial_loss', 0) for s in early])
    mid_gen = np.mean([s.get('gen_adversarial_loss', 0) for s in mid])
    late_gen = np.mean([s.get('gen_adversarial_loss', 0) for s in late])
    
    print()
    print("📈 Generator Adversarial Loss:")
    print(f"  Early (1-10): {early_gen:.4f}")
    print(f"  Mid: {mid_gen:.4f}")
    print(f"  Late: {late_gen:.4f}")
    
    gen_trend = "📈 INCREASING" if late_gen > early_gen else "📉 DECREASING"
    gen_change = abs(late_gen - early_gen)
    print(f"  Trend: {gen_trend} (change: {gen_change:.4f})")
    
    # Discriminator Health
    early_disc_real = np.mean([s.get('disc_real_loss', 0) for s in early])
    late_disc_real = np.mean([s.get('disc_real_loss', 0) for s in late])
    
    print()
    print("🛡️ Discriminator Health:")
    print(f"  Real Loss Early: {early_disc_real:.4f}")
    print(f"  Real Loss Late: {late_disc_real:.4f}")
    
    disc_health = "✅ Healthy" if 0.3 <= late_disc_real <= 0.8 else "❌ Unhealthy"
    print(f"  Status: {disc_health}")
    
    # Quality Scores
    early_gen_qual = np.mean([s.get('gen_quality', 0) for s in early])
    late_gen_qual = np.mean([s.get('gen_quality', 0) for s in late])
    early_nca_qual = np.mean([s.get('nca_quality', 0) for s in early])
    late_nca_qual = np.mean([s.get('nca_quality', 0) for s in late])
    
    print()
    print("⭐ Quality Scores:")
    print(f"  Gen Quality: {early_gen_qual:.3f} → {late_gen_qual:.3f}")
    print(f"  NCA Quality: {early_nca_qual:.3f} → {late_nca_qual:.3f}")
    
    # Overall Assessment
    print()
    print("🎯 LEARNING ASSESSMENT:")
    
    # Check if losses are stable/improving
    loss_stable = nca_change < 0.1 and gen_change < 0.1
    quality_improving = late_gen_qual > early_gen_qual or late_nca_qual > early_nca_qual
    disc_healthy = 0.3 <= late_disc_real <= 0.8
    
    if loss_stable and disc_healthy:
        print("✅ STABLE LEARNING: Losses are stable and discriminator is healthy")
    elif quality_improving:
        print("🔄 PROGRESSIVE LEARNING: Quality scores improving despite loss changes")
    elif not disc_healthy:
        print("⚠️ DISCRIMINATOR ISSUES: Need to rebalance learning rates")
    else:
        print("❌ LEARNING PROBLEMS: Losses increasing without quality improvement")
    
    # Learning rate recommendations based on GAN Hacks
    print()
    print("🔧 RECOMMENDATIONS (based on GAN training best practices):")
    
    if late_disc_real > 0.8:
        print("  • Increase discriminator learning rate")
        print("  • Decrease generator/NCA learning rates")
    elif late_disc_real < 0.3:
        print("  • Decrease discriminator learning rate") 
        print("  • Increase generator/NCA learning rates")
    
    if nca_change > 0.05 and gen_change > 0.05:
        print("  • Apply gradient clipping")
        print("  • Consider learning rate scheduling")
    
    if not quality_improving:
        print("  • Check if discriminator is providing meaningful feedback")
        print("  • Consider using different loss formulations")

if __name__ == "__main__":
    analyze_learning() 