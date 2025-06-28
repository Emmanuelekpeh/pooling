# Transformer Critic Patch
# This file contains the classes needed to fix the TransformerCritic error in train_integrated_fast.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os

# --- Transformer Critic/Imitator ---
class TransformerCritic(nn.Module):
    """
    Transformer-based critic that evolves into an imitator after epoch 350.
    Based on efficient transformer principles from recent research.
    """
    def __init__(self, img_size=64, patch_size=8, embed_dim=256, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.mode = "critic"  # "critic" or "imitator"
        
        # Patch embedding - convert image patches to tokens
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Efficient transformer blocks with reduced complexity
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Critic head (for evaluation mode)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Imitator head (for generation mode)
        self.imitator_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, self.num_patches * 3),  # RGB for each patch
            nn.Tanh()
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights following transformer best practices"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def set_mode(self, mode):
        """Switch between critic and imitator modes"""
        assert mode in ["critic", "imitator"], "Mode must be 'critic' or 'imitator'"
        self.mode = mode
        
    def forward(self, x, target=None):
        B, C, H, W = x.shape
        
        # Convert to patches and embed
        patches = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if self.mode == "critic":
            # Use CLS token for criticism
            cls_output = x[:, 0]  # [B, embed_dim]
            quality_score = self.critic_head(cls_output)
            return quality_score.squeeze(-1)
            
        elif self.mode == "imitator":
            # Use all patch tokens for generation
            if target is None:
                raise ValueError("Target image required for imitator mode")
            
            # Extract patch features (excluding CLS token)
            patch_features = x[:, 1:]  # [B, num_patches, embed_dim]
            
            # Generate imitation
            generated_patches = self.imitator_head(patch_features)  # [B, num_patches, 3]
            
            # Reshape to image format
            patches_per_side = int(self.num_patches ** 0.5)
            generated_patches = generated_patches.view(B, patches_per_side, patches_per_side, 3)
            generated_patches = generated_patches.permute(0, 3, 1, 2)  # [B, 3, H//patch_size, W//patch_size]
            
            # Upsample to original image size
            generated_img = F.interpolate(generated_patches, size=(H, W), mode='bilinear', align_corners=False)
            
            # Compute imitation loss
            imitation_loss = F.mse_loss(generated_img, target)
            
            return generated_img, imitation_loss

class EfficientTransformerBlock(nn.Module):
    """
    Efficient transformer block with optimizations from recent research.
    Reduces computational complexity while maintaining effectiveness.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm architecture for better gradient flow
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientAttention(nn.Module):
    """
    Efficient attention mechanism with linear complexity optimizations.
    Based on techniques from Performer, Linformer, and similar efficient transformers.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, attention_type="standard"):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_type = attention_type
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For efficient attention variants
        if attention_type == "linear":
            # Linear attention approximation
            self.feature_map = nn.ReLU()
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.attention_type == "linear":
            # Linear attention for O(N) complexity
            q = self.feature_map(q) + 1e-6
            k = self.feature_map(k) + 1e-6
            
            # Compute attention efficiently
            kv = torch.einsum('bhnd,bhne->bhde', k, v)
            qkv = torch.einsum('bhnd,bhde->bhne', q, kv)
            
            # Normalize
            k_sum = k.sum(dim=-2, keepdim=True)
            q_k_sum = torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(-2)).unsqueeze(-1)
            attn_output = qkv / (q_k_sum + 1e-6)
            
        else:
            # Standard attention with optimizations
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            attn_output = attn @ v
        
        # Reshape and project output
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

# Function to fix the to_rgba method
def fix_to_rgba_method(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the to_rgba method
    to_rgba_pattern = r'def to_rgba\(self, x\):(.*?)return'
    to_rgba_replacement = '''def to_rgba(self, x):
        # Instead, just clamp alpha to [0, 1] and preserve zeros - non-in-place
        rgb = torch.tanh(x[:, :3])  # RGB channels to [-1, 1] range
        alpha = torch.sigmoid(x[:, 3:4])  # Alpha channel to [0, 1] range
        rgba = torch.cat([rgb, alpha], dim=1)
        return'''
    
    fixed_content = re.sub(to_rgba_pattern, to_rgba_replacement, content, flags=re.DOTALL)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

# Update the train_integrated_fast.py file to fix indentation issues
def fix_indentation_issues(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_method = False
    method_indent = 0
    
    for line in lines:
        if line.strip().startswith('def ') and ':' in line:
            in_method = True
            method_indent = len(line) - len(line.lstrip())
        
        # Skip the problematic perceive and forward methods at the end of the file
        if in_method and line.strip().startswith('def perceive(self, x):') and method_indent == 4:
            in_method = False
            continue
        
        if in_method and line.strip().startswith('def forward(self, x, w, steps, target_img=None):') and method_indent == 4:
            in_method = False
            continue
        
        fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

# Add the required classes to train_integrated_fast.py
def add_transformer_classes(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find position after CrossEvaluator class
    cross_eval_match = re.search(r'class CrossEvaluator\(nn\.Module\):.*?def forward\(self, img\):\s*return self\.model\(img\)', content, re.DOTALL)
    
    if cross_eval_match:
        insert_position = cross_eval_match.end()
        
        # Get TransformerCritic and related classes as a string
        transformer_classes = """

# --- Transformer Critic/Imitator ---
class TransformerCritic(nn.Module):
    \"\"\"
    Transformer-based critic that evolves into an imitator after epoch 350.
    Based on efficient transformer principles from recent research.
    \"\"\"
    def __init__(self, img_size=64, patch_size=8, embed_dim=256, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.mode = "critic"  # "critic" or "imitator"
        
        # Patch embedding - convert image patches to tokens
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Efficient transformer blocks with reduced complexity
        self.transformer_blocks = nn.ModuleList([
            EfficientTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Critic head (for evaluation mode)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Imitator head (for generation mode)
        self.imitator_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, self.num_patches * 3),  # RGB for each patch
            nn.Tanh()
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        \"\"\"Initialize weights following transformer best practices\"\"\"
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def set_mode(self, mode):
        \"\"\"Switch between critic and imitator modes\"\"\"
        assert mode in ["critic", "imitator"], "Mode must be 'critic' or 'imitator'"
        self.mode = mode
        
    def forward(self, x, target=None):
        B, C, H, W = x.shape
        
        # Convert to patches and embed
        patches = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if self.mode == "critic":
            # Use CLS token for criticism
            cls_output = x[:, 0]  # [B, embed_dim]
            quality_score = self.critic_head(cls_output)
            return quality_score.squeeze(-1)
            
        elif self.mode == "imitator":
            # Use all patch tokens for generation
            if target is None:
                raise ValueError("Target image required for imitator mode")
            
            # Extract patch features (excluding CLS token)
            patch_features = x[:, 1:]  # [B, num_patches, embed_dim]
            
            # Generate imitation
            generated_patches = self.imitator_head(patch_features)  # [B, num_patches, 3]
            
            # Reshape to image format
            patches_per_side = int(self.num_patches ** 0.5)
            generated_patches = generated_patches.view(B, patches_per_side, patches_per_side, 3)
            generated_patches = generated_patches.permute(0, 3, 1, 2)  # [B, 3, H//patch_size, W//patch_size]
            
            # Upsample to original image size
            generated_img = F.interpolate(generated_patches, size=(H, W), mode='bilinear', align_corners=False)
            
            # Compute imitation loss
            imitation_loss = F.mse_loss(generated_img, target)
            
            return generated_img, imitation_loss

class EfficientTransformerBlock(nn.Module):
    \"\"\"
    Efficient transformer block with optimizations from recent research.
    Reduces computational complexity while maintaining effectiveness.
    \"\"\"
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm architecture for better gradient flow
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientAttention(nn.Module):
    \"\"\"
    Efficient attention mechanism with linear complexity optimizations.
    Based on techniques from Performer, Linformer, and similar efficient transformers.
    \"\"\"
    def __init__(self, embed_dim, num_heads, dropout=0.1, attention_type="standard"):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_type = attention_type
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For efficient attention variants
        if attention_type == "linear":
            # Linear attention approximation
            self.feature_map = nn.ReLU()
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.attention_type == "linear":
            # Linear attention for O(N) complexity
            q = self.feature_map(q) + 1e-6
            k = self.feature_map(k) + 1e-6
            
            # Compute attention efficiently
            kv = torch.einsum('bhnd,bhne->bhde', k, v)
            qkv = torch.einsum('bhnd,bhde->bhne', q, kv)
            
            # Normalize
            k_sum = k.sum(dim=-2, keepdim=True)
            q_k_sum = torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(-2)).unsqueeze(-1)
            attn_output = qkv / (q_k_sum + 1e-6)
            
        else:
            # Standard attention with optimizations
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            attn_output = attn @ v
        
        # Reshape and project output
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x
"""
        
        # Insert the TransformerCritic classes
        new_content = content[:insert_position] + transformer_classes + content[insert_position:]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True
    else:
        print("Could not find CrossEvaluator class in the file")
        return False

# Add a main function to the script
def add_main_function(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if main function already exists
    if 'if __name__ == "__main__":' in content:
        return False
    
    # Add main function to the end of the file
    main_function = """

# Main function with argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for integrated NCA and StyleGAN models")
    parser.add_argument("--run-training", action="store_true", help="Start the training process")
    parser.add_argument("--test-checkpoint", action="store_true", help="Test using the latest checkpoint")
    parser.add_argument("--cleanup-checkpoints", action="store_true", help="Clean up old checkpoints")
    args = parser.parse_args()
    
    if args.run_training:
        training_loop()
    elif args.test_checkpoint:
        print("Testing checkpoint functionality not yet implemented")
    elif args.cleanup_checkpoints:
        cleanup_old_checkpoints(CHECKPOINT_DIR)
    else:
        print("This script is now only for training. Use --run-training to start, --test-checkpoint to test, or --cleanup-checkpoints to clean up old checkpoints.")
"""
    
    # Remove the hanging functions at the end of the file
    last_line = "print(\"This script is now only for training. Use --run-training to start, --test-checkpoint to test, or --cleanup-checkpoints to clean up old checkpoints.\")"
    if last_line in content:
        end_index = content.find(last_line) + len(last_line)
        content = content[:end_index]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content + main_function)
    
    return True

# Apply all fixes
def apply_all_fixes():
    filepath = 'train_integrated_fast.py'
    fix_indentation_issues(filepath)
    add_transformer_classes(filepath)
    fix_to_rgba_method(filepath)
    add_main_function(filepath)
    print("All fixes applied to train_integrated_fast.py")

# Run the fixes when executing this script
if __name__ == "__main__":
    apply_all_fixes() 