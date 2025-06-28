import torch
import os
import json
import random

def create_sample_checkpoint():
    """Create a sample checkpoint with training data to test the line charts"""
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Creating checkpoint directory at: {os.path.abspath(checkpoint_dir)}")
    
    # Generate sample training data (simulating 10 epochs)
    sample_scores = []
    
    for epoch in range(1, 11):
        # Simulate realistic training metrics that improve over time
        base_quality = 0.3 + (epoch / 10) * 0.4  # Quality improves from 0.3 to 0.7
        noise = random.uniform(-0.05, 0.05)      # Add some noise
        
        epoch_data = {
            # Losses (should generally decrease)
            'gen_eval_loss': max(0.1, 1.0 - epoch * 0.08 + noise),
            'nca_eval_loss': max(0.1, 0.8 - epoch * 0.06 + noise),
            'gen_penalty': max(0.05, 0.5 - epoch * 0.03 + noise),
            'nca_penalty': max(0.05, 0.6 - epoch * 0.04 + noise),
            'feature_match_gen': max(0.02, 0.3 - epoch * 0.02 + noise),
            'feature_match_nca': max(0.02, 0.4 - epoch * 0.03 + noise),
            'transformer_loss': max(0.1, 0.7 - epoch * 0.05 + noise),
            
            # Quality scores (should generally increase)
            'gen_quality': min(0.9, base_quality + noise),
            'nca_quality': min(0.9, base_quality - 0.1 + noise),
            'ensemble_prediction': min(0.9, base_quality + 0.05 + noise),
            'cross_learning_loss': max(0.05, 0.4 - epoch * 0.02 + noise)
        }
        
        sample_scores.append(epoch_data)
    
    # Create a sample checkpoint structure
    checkpoint = {
        'epoch': 10,
        'scores': sample_scores,
        # Add minimal dummy states for required models and optimizers
        'generator_state_dict': {},
        'nca_state_dict': {},
        'discriminator_state_dict': {},
        'gen_evaluator_state_dict': {},
        'nca_evaluator_state_dict': {},
        'transformer_critic_state_dict': {},
        'transformer_mode': 'critic',
        'cross_learning_system_state_dict': {},
        'opt_gen_state_dict': {},
        'opt_nca_state_dict': {},
        'opt_disc_state_dict': {},
        'opt_gen_eval_state_dict': {},
        'opt_nca_eval_state_dict': {},
        'opt_transformer_state_dict': {},
        'opt_cross_learning_state_dict': {},
        'losses': []
    }
    
    # Save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Created sample checkpoint at {os.path.abspath(checkpoint_path)}")
        print(f"   - Contains {len(sample_scores)} epochs of training data")
        
        # Verify the file exists and has expected size
        if os.path.exists(checkpoint_path):
            print(f"   - File size: {os.path.getsize(checkpoint_path)} bytes")
            # Try to load it back to ensure it was saved correctly
            try:
                loaded = torch.load(checkpoint_path)
                print(f"   - Successfully loaded the checkpoint")
                print(f"   - Contains keys: {list(loaded.keys())}")
                print(f"   - Scores data length: {len(loaded.get('scores', []))}")
            except Exception as e:
                print(f"❌ Error loading checkpoint: {str(e)}")
        else:
            print(f"❌ File was not created: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error saving checkpoint: {str(e)}")
        # Print directory permissions
        try:
            print(f"Directory permissions: {oct(os.stat(checkpoint_dir).st_mode & 0o777)}")
        except:
            pass

def update_checkpoint():
    """Update the checkpoint with one more epoch of data"""
    checkpoint_path = os.path.join("./checkpoints", "latest_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found. Create one first.")
        return
        
    # Load existing checkpoint
    try:
        checkpoint = torch.load(checkpoint_path)
        scores = checkpoint.get('scores', [])
        
        # Get the current epoch
        current_epoch = len(scores) + 1
        
        # Generate new epoch data
        base_quality = min(0.9, 0.3 + (current_epoch / 10) * 0.4)
        noise = random.uniform(-0.05, 0.05)
        
        new_epoch = {
            'gen_eval_loss': max(0.1, 1.0 - current_epoch * 0.08 + noise),
            'nca_eval_loss': max(0.1, 0.8 - current_epoch * 0.06 + noise),
            'gen_penalty': max(0.05, 0.5 - current_epoch * 0.03 + noise),
            'nca_penalty': max(0.05, 0.6 - current_epoch * 0.04 + noise),
            'feature_match_gen': max(0.02, 0.3 - current_epoch * 0.02 + noise),
            'feature_match_nca': max(0.02, 0.4 - current_epoch * 0.03 + noise),
            'transformer_loss': max(0.1, 0.7 - current_epoch * 0.05 + noise),
            'gen_quality': min(0.9, base_quality + noise),
            'nca_quality': min(0.9, base_quality - 0.1 + noise),
            'ensemble_prediction': min(0.9, base_quality + 0.05 + noise),
            'cross_learning_loss': max(0.05, 0.4 - current_epoch * 0.02 + noise)
        }
        
        # Add new epoch to scores
        scores.append(new_epoch)
        checkpoint['scores'] = scores
        checkpoint['epoch'] = current_epoch
        
        # Save updated checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Updated checkpoint with epoch {current_epoch}")
        print(f"   - Now contains {len(scores)} epochs of data")
    except Exception as e:
        print(f"❌ Error updating checkpoint: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        update_checkpoint()
    else:
        create_sample_checkpoint()
        
    print(f"\n🎯 Now refresh your browser at http://localhost:5000 to see the line charts!")
    print(f"📈 You should see:")
    print(f"   • Training Losses chart with 4 lines showing decreasing losses")
    print(f"   • Quality Scores chart with 4 lines showing improving metrics")
    print(f"\n💡 Run 'python test_chart_data.py update' to add one more epoch of data") 