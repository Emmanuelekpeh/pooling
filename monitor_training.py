#!/usr/bin/env python3
"""
Simple script to monitor the training progress and check for Ghost Memory system activity.
"""

import os
import time
import psutil
import subprocess

def check_training_status():
    print("🔍 Monitoring Training Status")
    print("=" * 50)
    
    # Check for Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if python_processes:
        print(f"📊 Found {len(python_processes)} Python process(es):")
        for proc in python_processes:
            try:
                cpu = proc.cpu_percent(interval=1)
                mem_mb = proc.memory_info().rss / 1024 / 1024
                print(f"  PID {proc.pid}: CPU {cpu:.1f}%, Memory {mem_mb:.1f}MB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"  PID {proc.pid}: Process no longer accessible")
    else:
        print("❌ No Python processes found")
    
    # Check checkpoint directory
    checkpoint_dir = "./checkpoints"
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        if files:
            print(f"\n💾 Checkpoints found: {len(files)} files")
            for f in files:
                path = os.path.join(checkpoint_dir, f)
                if os.path.isfile(path):
                    size_mb = os.path.getsize(path) / 1024 / 1024
                    mtime = os.path.getmtime(path)
                    time_str = time.strftime('%H:%M:%S', time.localtime(mtime))
                    print(f"  {f}: {size_mb:.1f}MB (modified {time_str})")
        else:
            print(f"\n💾 Checkpoint directory exists but is empty")
    else:
        print(f"\n❌ Checkpoint directory not found")
    
    # Check for any output files or logs
    output_files = [f for f in os.listdir('.') if f.endswith(('.log', '.txt', '.out'))]
    if output_files:
        print(f"\n📝 Output files found: {output_files}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_training_status() 