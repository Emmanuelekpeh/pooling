#!/usr/bin/env python3
"""
Fast training script for quick iteration using smaller dataset
"""
import subprocess
import sys
import os

def modify_config_for_fast_training():
    """Modify the training configuration for faster iteration"""
    
    # Read the current training script
    with open('train_integrated.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Make modifications for faster training
    modifications = [
        # Use smaller dataset
        ('DATA_DIR = "./data/ukiyo-e"', 'DATA_DIR = "./data/ukiyo-e-small"'),
        # Reduce epochs for quick testing
        ('EPOCHS = 300', 'EPOCHS = 10'),
        # Reduce NCA steps for faster forward passes
        ('NCA_STEPS_MIN = 64', 'NCA_STEPS_MIN = 32'),
        ('NCA_STEPS_MAX = 96', 'NCA_STEPS_MAX = 48'),
    ]
    
    modified_content = content
    for old, new in modifications:
        if old in modified_content:
            modified_content = modified_content.replace(old, new)
            print(f"✓ Modified: {old} -> {new}")
        else:
            print(f"⚠ Warning: Could not find: {old}")
    
    # Write the modified content to a new file
    with open('train_integrated_fast.py', 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"\n✓ Created fast training script: train_integrated_fast.py")
    print(f"  - Dataset: 19 images (vs 166)")
    print(f"  - Epochs: 10 (vs 300)")
    print(f"  - NCA Steps: 32-48 (vs 64-96)")

def run_fast_training():
    """Run the fast training"""
    print("\n🚀 Starting fast training...")
    
    # Kill any existing training processes first
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      capture_output=True, text=True)
        print("✓ Stopped existing Python processes")
    except:
        pass
    
    # Run the fast training
    cmd = [sys.executable, 'train_integrated_fast.py', '--run-training']
    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    print("✓ Started fast training in new console window")

if __name__ == "__main__":
    modify_config_for_fast_training()
    
    # Ask user if they want to start training
    response = input("\nDo you want to start fast training now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_fast_training()
    else:
        print("Fast training script created. Run with: python train_integrated_fast.py --run-training") 