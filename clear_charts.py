import torch
import os

def clear_chart_data():
    """Clear all chart data by creating a checkpoint with empty scores."""
    # Define checkpoint directory (use same as in train_integrated_fast.py)
    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints" if os.path.exists("/app") else "./checkpoints")
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Path to latest checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
    
    # If a checkpoint exists, load it to preserve model state
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            # Clear scores data
            checkpoint['scores'] = []
            torch.save(checkpoint, checkpoint_path)
            print(f"Successfully cleared scores data from existing checkpoint at {checkpoint_path}")
        except Exception as e:
            print(f"Error modifying existing checkpoint: {e}")
    else:
        # Create a minimal empty checkpoint with no scores
        empty_checkpoint = {
            'epoch': 0,
            'scores': []
        }
        torch.save(empty_checkpoint, checkpoint_path)
        print(f"Created new checkpoint with empty scores at {checkpoint_path}")
    
    print("Chart data has been cleared. Restart or refresh the web interface to see the changes.")

if __name__ == "__main__":
    clear_chart_data() 