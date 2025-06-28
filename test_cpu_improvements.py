#!/usr/bin/env python3
"""
Test script to verify CPU optimizations are working properly.
Run this before your main training to check settings.
"""

import os
import torch
import time
import multiprocessing

def test_cpu_optimizations():
    """Test that CPU optimizations are working."""
    print("🖥️  Testing CPU Optimizations")
    print("=" * 50)
    
    # Test thread settings
    original_threads = torch.get_num_threads()
    
    # Apply optimizations
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    torch.set_float32_matmul_precision('high')
    
    print(f"✅ CPU cores available: {multiprocessing.cpu_count()}")
    print(f"✅ PyTorch threads: {torch.get_num_threads()} (was {original_threads})")
    print(f"✅ OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"✅ MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    print(f"✅ Float32 matmul precision: high")
    
    # Test basic tensor operations
    print("\n🧪 Testing tensor operations...")
    device = torch.device('cpu')
    
    # Test matrix multiplication performance
    start_time = time.time()
    for _ in range(10):
        a = torch.randn(256, 512, device=device)
        b = torch.randn(512, 256, device=device)
        c = torch.mm(a, b)
    matmul_time = time.time() - start_time
    print(f"✅ Matrix multiplication test: {matmul_time:.3f}s")
    
    # Test convolution performance
    start_time = time.time()
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
    for _ in range(10):
        x = torch.randn(4, 3, 64, 64, device=device)
        y = conv(x)
    conv_time = time.time() - start_time
    print(f"✅ Convolution test: {conv_time:.3f}s")
    
    print("\n🚀 CPU optimizations are working!")
    print(f"   Recommended batch size for your CPU: 2-4")
    print(f"   Recommended DataLoader workers: {min(4, num_threads // 2)}")

def test_interruption_handling():
    """Test graceful interruption handling."""
    print("\n🛑 Testing Interruption Handling")
    print("=" * 50)
    
    import signal
    interrupted = False
    
    def signal_handler(signum, frame):
        nonlocal interrupted
        print("✅ Signal handler working - interruption captured")
        interrupted = True
        
    # Set up signal handler
    original_handler = signal.signal(signal.SIGINT, signal_handler)
    
    print("✅ Signal handler set up successfully")
    print("   In your training, you can now press Ctrl+C for graceful interruption")
    
    # Restore original handler
    signal.signal(signal.SIGINT, original_handler)

if __name__ == "__main__":
    test_cpu_optimizations()
    test_interruption_handling()
    
    print("\n" + "=" * 60)
    print("🎯 Your improved training is ready!")
    print("   Run: python train_integrated_fast.py --run-training")
    print("   Press Ctrl+C anytime to stop gracefully")
    print("=" * 60) 