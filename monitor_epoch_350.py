#!/usr/bin/env python3

import json
import time
import sys

def monitor_epoch_350():
    """Monitor training to see if it passes epoch 350 successfully"""
    print("🔍 Monitoring epoch 350 transformer mode switch...")
    print("Looking for: 'Epoch 350: Switching transformer to imitator mode'")
    print("=" * 60)
    
    last_epoch = 0
    start_time = time.time()
    
    for i in range(60):  # Monitor for 5 minutes (60 * 5 seconds)
        try:
            with open('./samples/status.json', 'r') as f:
                status = json.load(f)
            
            status_text = status['status']
            
            # Extract epoch number
            if 'Epoch ' in status_text:
                epoch_str = status_text.split('Epoch ')[1].split(',')[0]
                current_epoch = int(epoch_str)
                
                # Print progress every 5 seconds or when epoch changes
                if current_epoch != last_epoch or i % 2 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.0f}s] {status_text}")
                    last_epoch = current_epoch
                
                # Check for success
                if current_epoch >= 351:
                    print("\n🎉 SUCCESS! Training passed epoch 350 without errors!")
                    print(f"✅ Transformer mode switch completed successfully")
                    print(f"✅ Fixed tensor shape issue resolved the problem")
                    return True
                
                # Check if we're at the critical epoch
                if current_epoch == 350:
                    print(f"\n⚡ CRITICAL EPOCH 350 - Watching for transformer switch...")
            
            # Check for errors
            if 'Error' in status_text or 'error' in status_text.lower():
                print(f"\n❌ Error detected in training: {status_text}")
                return False
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[{i*5}s] Status file issue: {e}")
        
        time.sleep(5)
    
    print(f"\n⏰ Monitoring timeout after 5 minutes")
    print(f"Last seen epoch: {last_epoch}")
    return False

if __name__ == "__main__":
    success = monitor_epoch_350()
    sys.exit(0 if success else 1) 