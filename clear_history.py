import torch
import os

CHECKPOINT_PATH = './checkpoints/latest_checkpoint.pt'

def clear_history():
    """
    Loads the latest checkpoint, clears the metrics_history,
    and saves it back.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Nothing to do.")
        return

    try:
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH)

        if 'metrics_history' in checkpoint and checkpoint['metrics_history']:
            print(f"Found metrics history with {len(checkpoint['metrics_history'])} epochs.")
            checkpoint['metrics_history'] = []
            torch.save(checkpoint, CHECKPOINT_PATH)
            print("✅ Metrics history cleared successfully.")
        else:
            print("No metrics history found in the checkpoint. Nothing to clear.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    clear_history() 