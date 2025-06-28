#!/usr/bin/env python3
"""
Test script to verify checkpoint saving and chart data functionality.
Run this to check if your charts will work properly.
"""

import os
import torch
import json
from train_integrated_fast import update_status

def test_checkpoint_functionality():
    """Test that checkpoint saving and loading works with metrics."""
    print("🧪 Testing Checkpoint & Chart Data Functionality")
    print("=" * 60)
    
    # Simulate some training metrics
    print("📊 Simulating training metrics...")
    
    # Create fake metrics like real training would
    fake_metrics = [
        {
            'gen_eval_loss': 0.8, 'nca_eval_loss': 0.7, 'gen_penalty': 0.5,
            'nca_penalty': 0.4, 'feature_match_gen': 0.3, 'feature_match_nca': 0.35,
            'transformer_loss': 0.25, 'gen_quality': 0.6, 'nca_quality': 0.65,
            'ensemble_prediction': 0.62, 'cross_learning_loss': 0.2
        },
        {
            'gen_eval_loss': 0.75, 'nca_eval_loss': 0.65, 'gen_penalty': 0.45,
            'nca_penalty': 0.35, 'feature_match_gen': 0.25, 'feature_match_nca': 0.3,
            'transformer_loss': 0.2, 'gen_quality': 0.7, 'nca_quality': 0.75,
            'ensemble_prediction': 0.72, 'cross_learning_loss': 0.15
        }
    ]
    
    # Simulate update_status calls with metrics
    for i, metrics in enumerate(fake_metrics):
        scores = {
            'disc_loss': metrics['gen_eval_loss'],
            'nca_loss': metrics['nca_eval_loss'], 
            'gen_loss': metrics['gen_penalty'],
            'transformer_loss': metrics['transformer_loss']
        }
        update_status(f"Testing epoch {i+1}", scores=scores)
        print(f"✅ Added metrics for epoch {i+1}")
    
    # Check metrics history
    if hasattr(update_status, 'metrics_history'):
        print(f"📊 Metrics history contains {len(update_status.metrics_history)} epochs")
        print(f"    Sample metrics: {list(update_status.metrics_history[0].keys())}")
    else:
        print("❌ No metrics history found!")
        return False
    
    # Test checkpoint saving format
    print("\n💾 Testing checkpoint format...")
    
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    test_checkpoint = {
        'epoch': 20,
        'generator_state_dict': {},  # Empty for test
        'discriminator_state_dict': {},
        'scores': getattr(update_status, 'metrics_history', [])
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, 'test_checkpoint.pt')
    torch.save(test_checkpoint, checkpoint_path)
    print(f"✅ Saved test checkpoint with {len(test_checkpoint['scores'])} metrics")
    
    # Test loading
    loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
    if 'scores' in loaded_checkpoint and loaded_checkpoint['scores']:
        print(f"✅ Loaded checkpoint with {len(loaded_checkpoint['scores'])} metrics")
        print(f"    First metric keys: {list(loaded_checkpoint['scores'][0].keys())}")
    else:
        print("❌ Loaded checkpoint missing scores data!")
        return False
    
    # Test chart data format
    print("\n📈 Testing chart data format...")
    try:
        # This simulates what the web server does
        chart_data = {
            'epochs': len(loaded_checkpoint['scores']),
            'metrics': loaded_checkpoint['scores']
        }
        
        # Verify it's JSON serializable
        json_str = json.dumps(chart_data)
        print(f"✅ Chart data is JSON serializable ({len(json_str)} chars)")
        
        # Check specific metrics exist
        required_metrics = ['gen_quality', 'nca_quality', 'transformer_loss']
        for metric in required_metrics:
            if metric in loaded_checkpoint['scores'][0]:
                print(f"✅ Found required metric: {metric}")
            else:
                print(f"❌ Missing metric: {metric}")
                
    except Exception as e:
        print(f"❌ Chart data format error: {e}")
        return False
    
    # Clean up
    os.remove(checkpoint_path)
    print(f"\n🧹 Cleaned up test files")
    
    return True

def test_server_compatibility():
    """Test that the data format works with the web server."""
    print("\n🌐 Testing Server Compatibility")
    print("=" * 40)
    
    # Check if samples directory and status.json exist
    samples_dir = "./samples"
    status_file = os.path.join(samples_dir, "status.json")
    
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            print(f"✅ Found status.json with keys: {list(status_data.keys())}")
            
            if 'scores' in status_data and status_data['scores']:
                print(f"✅ Status contains scores data")
            else:
                print("⚠️  Status.json exists but no scores data")
                
        except Exception as e:
            print(f"❌ Error reading status.json: {e}")
    else:
        print("ℹ️  No status.json found (normal for fresh start)")
    
    # Test checkpoint directory
    checkpoint_dir = "./checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        print(f"✅ Found {len(checkpoints)} checkpoint files")
        
        if 'latest_checkpoint.pt' in checkpoints:
            print("✅ latest_checkpoint.pt exists")
            try:
                checkpoint = torch.load(os.path.join(checkpoint_dir, 'latest_checkpoint.pt'), weights_only=False)
                if 'scores' in checkpoint:
                    print(f"✅ Latest checkpoint contains {len(checkpoint.get('scores', []))} metrics")
                else:
                    print("⚠️  Latest checkpoint missing scores data")
            except Exception as e:
                print(f"❌ Error loading latest checkpoint: {e}")
    else:
        print("ℹ️  No checkpoints directory found")

if __name__ == "__main__":
    print("🔍 Checkpoint & Chart Data Test Suite")
    print("=" * 60)
    
    success = test_checkpoint_functionality()
    test_server_compatibility()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Your charts should work now.")
        print("   Run training and check the web interface at http://localhost:5000")
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("=" * 60) 