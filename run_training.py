#!/usr/bin/env python3
import argparse
import importlib.util
import sys

def import_module_from_file(module_name, file_path):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description="Training script for integrated NCA and StyleGAN models")
    parser.add_argument("--run-training", action="store_true", help="Start the training process")
    parser.add_argument("--test-checkpoint", action="store_true", help="Test using the latest checkpoint")
    parser.add_argument("--cleanup-checkpoints", action="store_true", help="Clean up old checkpoints")
    args = parser.parse_args()
    
    # Import the training module
    train_module = import_module_from_file("train_integrated_fast", "train_integrated_fast.py")
    
    if args.run_training:
        train_module.training_loop()
    elif args.test_checkpoint:
        print("Testing checkpoint functionality not yet implemented")
    elif args.cleanup_checkpoints:
        train_module.cleanup_old_checkpoints(train_module.CHECKPOINT_DIR)
    else:
        print("This script is for training. Use --run-training to start, --test-checkpoint to test, or --cleanup-checkpoints to clean up old checkpoints.")

if __name__ == "__main__":
    main() 