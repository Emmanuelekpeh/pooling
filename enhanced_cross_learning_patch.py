import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Copy these classes to train_integrated_fast.py after the EfficientAttention class

class SignalWeightingNetwork(nn.Module):
    """
    Adaptive signal weighting system that learns optimal weights for combining
    multiple evaluation signals from different models.
    """
    def __init__(self, num_signals=5, hidden_dim=64):
        super().__init__()
        self.num_signals = num_signals
        
        # Context encoder: learns to understand the current training state
        self.context_encoder = nn.Sequential(
            nn.Linear(num_signals + 3, hidden_dim),  # +3 for epoch, batch, time context
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_signals),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )
        
        # Signal confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_signals, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_signals),
            nn.Sigmoid()  # Confidence scores [0,1]
        )
        
        # Performance tracker for adaptive weighting
        self.register_buffer('signal_performance_history', torch.zeros(num_signals, 100))  # Last 100 batches
        self.register_buffer('history_index', torch.tensor(0))
        
    def update_performance_history(self, signal_scores, actual_performance):
        """Update performance history for each signal"""
        with torch.no_grad():
            idx = self.history_index.item() % 100
            # Calculate how well each signal predicted actual performance
            # signal_scores: [num_signals, batch_size], actual_performance: [batch_size]
            signal_accuracy = 1.0 - torch.abs(signal_scores - actual_performance.unsqueeze(0))
            # Take mean across batch dimension and flatten to [num_signals]
            accuracy_per_signal = signal_accuracy.mean(dim=1).squeeze()
            if accuracy_per_signal.dim() == 0:  # Handle single signal case
                accuracy_per_signal = accuracy_per_signal.unsqueeze(0)
            self.signal_performance_history[:, idx] = accuracy_per_signal
            self.history_index += 1
    
    def forward(self, signals, epoch_progress, batch_progress):
        """
        Args:
            signals: [num_signals, batch_size] - scores from different evaluators
            epoch_progress: float - progress through current epoch [0,1]
            batch_progress: float - progress through training [0,1]
        """
        batch_size = signals.shape[1]
        
        # Create context vector - ensure it's [batch_size, 3]
        context = torch.tensor([
            epoch_progress,
            batch_progress,
            signals.std().item()  # Signal diversity as context
        ], device=signals.device)
        # Reshape to [1, 3] then repeat to [batch_size, 3]
        context = context.view(1, -1).repeat(batch_size, 1)
        
        # Combine signals with context for weight prediction
        # signals.mean(dim=1) gives [num_signals], we want [batch_size, num_signals]
        signal_means = signals.mean(dim=1)  # [num_signals]
        signal_means = signal_means.view(1, -1).repeat(batch_size, 1)  # [batch_size, num_signals]
        weight_input = torch.cat([signal_means, context], dim=1)
        
        # Get adaptive weights
        adaptive_weights = self.context_encoder(weight_input)  # [batch_size, num_signals]
        
        # Get confidence scores
        confidence_scores = self.confidence_estimator(signal_means)  # [batch_size, num_signals]
        
        # Historical performance weighting
        if self.history_index > 10:  # Only after some history
            historical_performance = self.signal_performance_history.mean(dim=1)  # [num_signals]
            historical_weights = F.softmax(historical_performance * 2.0, dim=0)  # Sharpen distribution
            # historical_weights is [num_signals], we want [batch_size, num_signals]
            historical_weights = historical_weights.view(1, -1).repeat(batch_size, 1)
        else:
            historical_weights = torch.ones_like(adaptive_weights) / self.num_signals
        
        # Combine adaptive weights with historical performance and confidence
        final_weights = (
            0.4 * adaptive_weights +           # 40% current context
            0.3 * historical_weights +         # 30% historical performance  
            0.3 * confidence_scores            # 30% confidence
        )
        
        # Normalize to sum to 1
        final_weights = F.softmax(final_weights * 2.0, dim=1)  # Sharpen
        
        # Apply weights to signals
        weighted_signals = (signals.mT * final_weights).sum(dim=1)  # [batch_size]
        
        return weighted_signals, final_weights, confidence_scores

class EnhancedCrossLearningSystem(nn.Module):
    """
    Enhanced system that enables all models to learn from each other's outputs
    with sophisticated signal weighting and cross-model knowledge transfer.
    """
    def __init__(self, img_size=64, num_models=5):
        super().__init__()
        self.num_models = num_models
        
        # Signal weighting networks for different learning objectives
        self.quality_weighter = SignalWeightingNetwork(num_signals=num_models)
        self.style_weighter = SignalWeightingNetwork(num_signals=num_models)
        self.content_weighter = SignalWeightingNetwork(num_signals=num_models)
        
        # Cross-model feature extractors
        self.feature_extractors = nn.ModuleDict({
            'discriminator': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 64, 128)
            ),
            'generator': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 64, 128)
            ),
            'nca': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 64, 128)
            ),
            'transformer': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 64, 128)
            )
        })
        
        # Cross-model learning networks
        self.cross_learners = nn.ModuleDict({
            'gen_from_nca': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'nca_from_gen': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'both_from_transformer': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # Outputs for both gen and nca
                nn.Sigmoid()
            ),
            'all_from_discriminator': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # Outputs for gen, nca, transformer
                nn.Sigmoid()
            )
        })
        
        # Ensemble predictor that combines all signals
        self.ensemble_predictor = nn.Sequential(
            nn.Linear(num_models, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def extract_cross_features(self, images_dict):
        """Extract features from images using cross-model feature extractors"""
        features = {}
        for model_name, images in images_dict.items():
            if model_name in self.feature_extractors:
                features[model_name] = self.feature_extractors[model_name](images)
        return features
    
    def compute_cross_learning_signals(self, features):
        """Compute cross-learning signals between models"""
        signals = {}
        
        # Generator learns from NCA
        if 'nca' in features:
            signals['gen_from_nca'] = self.cross_learners['gen_from_nca'](features['nca'])
        
        # NCA learns from Generator  
        if 'generator' in features:
            signals['nca_from_gen'] = self.cross_learners['nca_from_gen'](features['generator'])
        
        # Both learn from Transformer
        if 'transformer' in features:
            transformer_signals = self.cross_learners['both_from_transformer'](features['transformer'])
            signals['gen_from_transformer'] = transformer_signals[:, 0:1]
            signals['nca_from_transformer'] = transformer_signals[:, 1:2]
        
        # All learn from Discriminator
        if 'discriminator' in features:
            disc_signals = self.cross_learners['all_from_discriminator'](features['discriminator'])
            signals['gen_from_discriminator'] = disc_signals[:, 0:1]
            signals['nca_from_discriminator'] = disc_signals[:, 1:2]
            signals['transformer_from_discriminator'] = disc_signals[:, 2:3]
        
        return signals
    
    def forward(self, images_dict, epoch_progress, batch_progress, target_performance=None):
        """
        Args:
            images_dict: {'real': tensor, 'generator': tensor, 'nca': tensor, 'transformer': tensor}
            epoch_progress: float [0,1]
            batch_progress: float [0,1]
            target_performance: ground truth performance for updating weights
        """
        # Extract cross-model features
        features = self.extract_cross_features(images_dict)
        
        # Compute cross-learning signals
        cross_signals = self.compute_cross_learning_signals(features)
        
        # Organize signals by target model
        gen_signals = torch.stack([
            cross_signals.get('gen_from_nca', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            cross_signals.get('gen_from_transformer', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            cross_signals.get('gen_from_discriminator', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            torch.ones_like(cross_signals[list(cross_signals.keys())[0]]) * 0.5,  # Baseline
            torch.ones_like(cross_signals[list(cross_signals.keys())[0]]) * 0.5   # Baseline
        ], dim=0)  # [5, batch_size]
        
        nca_signals = torch.stack([
            cross_signals.get('nca_from_gen', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            cross_signals.get('nca_from_transformer', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            cross_signals.get('nca_from_discriminator', torch.zeros_like(cross_signals[list(cross_signals.keys())[0]])),
            torch.ones_like(cross_signals[list(cross_signals.keys())[0]]) * 0.5,  # Baseline
            torch.ones_like(cross_signals[list(cross_signals.keys())[0]]) * 0.5   # Baseline
        ], dim=0)  # [5, batch_size]
        
        # Apply signal weighting
        weighted_gen_quality, gen_weights, gen_confidence = self.quality_weighter(
            gen_signals, epoch_progress, batch_progress
        )
        weighted_nca_quality, nca_weights, nca_confidence = self.quality_weighter(
            nca_signals, epoch_progress, batch_progress
        )
        
        # Ensemble prediction - ensure all tensors have the same shape [batch_size]
        transformer_signal = cross_signals.get('transformer_from_discriminator', torch.zeros_like(weighted_gen_quality))
        # Ensure transformer_signal is [batch_size] shape
        if transformer_signal.dim() > 1:
            transformer_signal = transformer_signal.squeeze()
        if transformer_signal.dim() == 0:  # scalar
            transformer_signal = transformer_signal.expand_as(weighted_gen_quality)
        
        all_signals = torch.stack([
            weighted_gen_quality,
            weighted_nca_quality,
            transformer_signal,
            torch.ones_like(weighted_gen_quality) * 0.5,  # Baseline
            torch.ones_like(weighted_gen_quality) * 0.5   # Baseline
        ], dim=1)  # [batch_size, 5]
        
        ensemble_prediction = self.ensemble_predictor(all_signals)
        
        # Update performance history if target provided
        if target_performance is not None:
            self.quality_weighter.update_performance_history(gen_signals, target_performance)
            self.quality_weighter.update_performance_history(nca_signals, target_performance)
        
        return {
            'weighted_gen_quality': weighted_gen_quality,
            'weighted_nca_quality': weighted_nca_quality,
            'ensemble_prediction': ensemble_prediction,
            'gen_weights': gen_weights,
            'nca_weights': nca_weights,
            'gen_confidence': gen_confidence,
            'nca_confidence': nca_confidence
        }

# To apply this patch, add these classes to train_integrated_fast.py at line 744 (after the EfficientAttention class)

def apply_patch():
    """Apply this patch to train_integrated_fast.py"""
    import re
    
    try:
        filepath = 'train_integrated_fast.py'
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find position after EfficientAttention class
        pattern = r'class EfficientAttention\(.*?\).*?def forward\(.*?\).*?return x\n'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            insert_position = match.end()
            
            # Get the classes as a string from this file
            with open(__file__, 'r', encoding='utf-8') as f:
                patch_content = f.read()
            
            # Extract the classes
            classes_pattern = r'class SignalWeightingNetwork.*?class EnhancedCrossLearningSystem.*?\'nca_confidence\': nca_confidence\n        \}'
            classes_match = re.search(classes_pattern, patch_content, re.DOTALL)
            
            if classes_match:
                classes_content = classes_match.group(0)
                
                # Insert the classes
                new_content = content[:insert_position] + '\n\n' + classes_content + '\n\n' + content[insert_position:]
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"Successfully added SignalWeightingNetwork and EnhancedCrossLearningSystem classes to {filepath}")
                return True
            else:
                print("Error: Could not extract classes from patch file")
                return False
        else:
            print("Error: Could not find insert position after EfficientAttention class")
            return False
    except Exception as e:
        print(f"Error applying patch: {str(e)}")
        return False

if __name__ == "__main__":
    apply_patch() 