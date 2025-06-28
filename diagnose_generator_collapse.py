#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os

def diagnose_generator_collapse():
    """Diagnose the root cause of generator collapse and suggest proper fixes"""
    
    print("🔍 Generator Collapse Diagnostic Tool")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_path = './checkpoints/latest_checkpoint.pt'
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found")
        return
    
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    print(f"📊 Analyzing checkpoint from epoch {checkpoint['epoch']}")
    
    # Load generator
    from train_integrated_fast import IntegratedGenerator
    generator = IntegratedGenerator(z_dim=64, w_dim=128)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print("\n🧬 Generator Architecture Analysis:")
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Generate diverse samples to analyze collapse
    print("\n🎲 Generating samples for collapse analysis...")
    samples = []
    latent_codes = []
    
    with torch.no_grad():
        for i in range(20):  # More samples for better analysis
            z = torch.randn(1, 64)
            img = generator(z)
            samples.append(img[0].cpu().numpy())
            latent_codes.append(z[0].cpu().numpy())
    
    samples_array = np.array(samples)
    latent_array = np.array(latent_codes)
    
    # Analyze output diversity
    print("\n📈 Output Diversity Analysis:")
    
    # 1. Value range analysis
    global_min = samples_array.min()
    global_max = samples_array.max()
    global_std = samples_array.std()
    global_mean = samples_array.mean()
    
    print(f"   Global range: [{global_min:.6f}, {global_max:.6f}]")
    print(f"   Global std: {global_std:.6f}")
    print(f"   Global mean: {global_mean:.6f}")
    
    # 2. Per-sample variance
    sample_variances = [np.var(sample) for sample in samples]
    avg_sample_variance = np.mean(sample_variances)
    print(f"   Average per-sample variance: {avg_sample_variance:.6f}")
    
    # 3. Inter-sample diversity
    sample_diffs = []
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            diff = np.mean(np.abs(samples[i] - samples[j]))
            sample_diffs.append(diff)
    
    avg_inter_sample_diff = np.mean(sample_diffs)
    print(f"   Average inter-sample difference: {avg_inter_sample_diff:.6f}")
    
    # 4. Unique value analysis
    all_values = samples_array.flatten()
    unique_values = len(np.unique(np.round(all_values, 4)))  # Round to 4 decimals
    total_values = len(all_values)
    uniqueness_ratio = unique_values / total_values
    
    print(f"   Unique values: {unique_values} / {total_values} ({uniqueness_ratio:.4f})")
    
    # 5. Channel analysis
    print("\n🎨 Channel Analysis:")
    for ch in range(3):
        ch_values = samples_array[:, ch, :, :].flatten()
        ch_std = np.std(ch_values)
        ch_range = np.max(ch_values) - np.min(ch_values)
        print(f"   Channel {ch}: std={ch_std:.6f}, range={ch_range:.6f}")
    
    # Diagnose the type of collapse
    print("\n🚨 Collapse Diagnosis:")
    
    collapse_score = 0
    issues = []
    
    if global_std < 0.01:
        collapse_score += 3
        issues.append("SEVERE: Extremely low output variance (mode collapse)")
    elif global_std < 0.05:
        collapse_score += 2
        issues.append("MODERATE: Low output variance")
    
    if avg_sample_variance < 0.001:
        collapse_score += 3
        issues.append("SEVERE: Samples are internally uniform (gray images)")
    elif avg_sample_variance < 0.01:
        collapse_score += 2
        issues.append("MODERATE: Low internal sample variance")
    
    if avg_inter_sample_diff < 0.01:
        collapse_score += 3
        issues.append("SEVERE: All samples are nearly identical")
    elif avg_inter_sample_diff < 0.05:
        collapse_score += 2
        issues.append("MODERATE: Low sample diversity")
    
    if uniqueness_ratio < 0.01:
        collapse_score += 3
        issues.append("SEVERE: Extremely limited value space")
    elif uniqueness_ratio < 0.1:
        collapse_score += 2
        issues.append("MODERATE: Limited value diversity")
    
    if global_max - global_min < 0.1:
        collapse_score += 2
        issues.append("MODERATE: Narrow output range")
    
    # Output diagnosis
    if collapse_score >= 8:
        severity = "🔴 CRITICAL"
    elif collapse_score >= 5:
        severity = "🟠 SEVERE"
    elif collapse_score >= 3:
        severity = "🟡 MODERATE"
    else:
        severity = "🟢 MILD"
    
    print(f"   Collapse Severity: {severity} (Score: {collapse_score}/15)")
    print("\n   Issues Detected:")
    for issue in issues:
        print(f"   - {issue}")
    
    # Suggest fixes based on research
    print("\n💡 Recommended Fixes (Based on AI Training Research):")
    
    if collapse_score >= 5:
        print("\n   🎯 IMMEDIATE ACTIONS:")
        print("   1. LEARNING RATE: Reduce generator LR by 50% (current training too aggressive)")
        print("   2. DISCRIMINATOR BALANCE: Train discriminator 2-3x more frequently")
        print("   3. NOISE INJECTION: Add noise to real images (prevent discriminator overfitting)")
        print("   4. LABEL SMOOTHING: Use 0.9 instead of 1.0 for real labels")
        print("   5. GRADIENT PENALTY: Add spectral normalization or gradient penalty")
    
    if avg_sample_variance < 0.001:
        print("\n   🎨 DIVERSITY FIXES:")
        print("   6. LATENT DIVERSITY: Increase latent space dimension or add noise")
        print("   7. ARCHITECTURE: Add skip connections or residual blocks")
        print("   8. LOSS FUNCTION: Add perceptual loss or feature matching")
    
    if uniqueness_ratio < 0.1:
        print("\n   🔧 TRAINING FIXES:")
        print("   9. BATCH SIZE: Reduce batch size for more gradient variance")
        print("   10. OPTIMIZER: Try Adam with different beta values or RMSprop")
        print("   11. REGULARIZATION: Add dropout or weight decay")
    
    print("\n   ⚠️  WHAT NOT TO DO:")
    print("   ❌ Manual color injection (doesn't fix underlying learning)")
    print("   ❌ Post-processing contrast enhancement (masks the problem)")
    print("   ❌ Increasing CFG scale without fixing training")
    
    # Create visualization
    print("\n📊 Creating diagnostic visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Show sample outputs
    for i in range(4):
        sample = samples[i].transpose(1, 2, 0)
        sample_normalized = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
        axes[0, i].imshow(sample_normalized)
        axes[0, i].set_title(f"Sample {i+1}\nVar: {sample_variances[i]:.6f}")
        axes[0, i].axis('off')
    
    # Show analysis plots
    axes[1, 0].hist(all_values, bins=50, alpha=0.7)
    axes[1, 0].set_title("Value Distribution")
    axes[1, 0].set_xlabel("Pixel Value")
    
    axes[1, 1].plot(sample_variances)
    axes[1, 1].set_title("Per-Sample Variance")
    axes[1, 1].set_xlabel("Sample Index")
    
    # Channel distributions
    colors = ['red', 'green', 'blue']
    for ch in range(3):
        ch_values = samples_array[:, ch, :, :].flatten()
        axes[1, 2].hist(ch_values, bins=30, alpha=0.5, color=colors[ch], label=f'Ch {ch}')
    axes[1, 2].set_title("Channel Distributions")
    axes[1, 2].legend()
    
    # Latent space analysis
    latent_std = np.std(latent_array, axis=0)
    axes[1, 3].plot(latent_std)
    axes[1, 3].set_title("Latent Space Std")
    axes[1, 3].set_xlabel("Latent Dimension")
    
    plt.tight_layout()
    plt.savefig('./generator_collapse_diagnosis.png', dpi=150, bbox_inches='tight')
    print("✅ Diagnostic visualization saved to: generator_collapse_diagnosis.png")
    
    # Save detailed report
    report = {
        'epoch': checkpoint['epoch'],
        'collapse_score': int(collapse_score),
        'severity': severity,
        'issues': issues,
        'metrics': {
            'global_std': float(global_std),
            'global_range': float(global_max - global_min),
            'avg_sample_variance': float(avg_sample_variance),
            'avg_inter_sample_diff': float(avg_inter_sample_diff),
            'uniqueness_ratio': float(uniqueness_ratio),
            'total_params': int(total_params)
        },
        'recommendations': [
            "Reduce generator learning rate by 50%",
            "Increase discriminator training frequency",
            "Add noise injection to real images",
            "Implement label smoothing",
            "Add gradient penalty or spectral normalization"
        ]
    }
    
    with open('./generator_collapse_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Detailed report saved to: generator_collapse_report.json")
    
    return report

if __name__ == "__main__":
    print("🔧 This tool identifies WHY the generator produces uniform outputs")
    print("🔧 Manual color injection is a band-aid - we need to fix the training")
    print()
    
    report = diagnose_generator_collapse()
    
    print("\n" + "=" * 60)
    print("✅ Diagnosis complete!")
    print("🔄 The real fix is in the training parameters, not post-processing")
    print("🔄 Check the generated report for specific recommendations") 