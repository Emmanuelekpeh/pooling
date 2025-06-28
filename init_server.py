import os
import json

# Create directories
os.makedirs('samples', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Initialize status.json
status = {
    'status': 'Server initialized',
    'images': [],
    'scores': {},
    'error': False
}

with open('samples/status.json', 'w') as f:
    json.dump(status, f)

print("Server directories and files initialized") 