import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalWeightingNetwork(nn.Module):
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
        
    def forward(self, signals, epoch_progress, batch_progress):
        """
        Args:
            signals: [num_signals, batch_size] tensor of quality signals
            epoch_progress: float [0,1] indicating progress through current epoch
            batch_progress: float [0,1] indicating overall training progress
        Returns:
            weighted_quality: [batch_size] tensor of weighted quality scores
            weights: [num_signals] tensor of signal weights
            confidence: [num_signals] tensor of confidence scores
        """
        # Prepare context vector
        batch_size = signals.shape[1]
        time_context = torch.tensor([epoch_progress, batch_progress, 0.5], 
                                  device=signals.device).expand(batch_size, 3)
        
        # Combine signals and context
        context_input = torch.cat([signals.t(), time_context], dim=1)
        
        # Get adaptive weights
        weights = self.context_encoder(context_input)  # [batch_size, num_signals]
        
        # Get confidence scores
        confidence = self.confidence_estimator(signals.t())  # [batch_size, num_signals]
        
        # Apply weighted sum
        weighted_signals = (signals.t() * weights * confidence)  # [batch_size, num_signals]
        weighted_quality = weighted_signals.sum(dim=1)  # [batch_size]
        
        return weighted_quality, weights, confidence

    def update_performance_history(self, signal_scores, actual_performance):
        """Update historical performance tracking for each signal"""
        idx = self.history_index.item()
        self.signal_performance_history[:, idx] = signal_scores.mean(dim=1)
        self.history_index[0] = (idx + 1) % 100

class EnhancedCrossLearningSystem(nn.Module):
    def __init__(self, img_size=64, num_models=5):
        super().__init__()
        self.num_models = num_models
        
        # Initialize feature extractors first
        self.feature_extractors = nn.ModuleDict({
            'discriminator': nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(8 * 8 * 128, 256)
            ),
            'generator': nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(8 * 8 * 128, 256)
            ),
            'nca': nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(8 * 8 * 128, 256)
            )
        })

        # Initialize signal weighting networks
        self.signal_weighting = nn.ModuleDict({
            'gen': SignalWeightingNetwork(num_signals=5),
            'nca': SignalWeightingNetwork(num_signals=5)
        })

        # Initialize ensemble predictor
        self.ensemble_predictor = nn.Sequential(
            nn.Linear(5, 32),  # 5 signals
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def extract_cross_features(self, images_dict):
        """Extract features from different model outputs using the feature extractors"""
        features = {}
        for model_name, extractor in self.feature_extractors.items():
            if model_name in images_dict:
                features[model_name] = extractor(images_dict[model_name])
        return features

    def compute_cross_learning_signals(self, features):
        """Compute cross-learning signals between models"""
        cross_signals = {}
        
        # Compare features between models
        if 'generator' in features and 'discriminator' in features:
            gen_disc_similarity = F.cosine_similarity(
                features['generator'],
                features['discriminator'],
                dim=1
            )
            cross_signals['gen_from_disc'] = gen_disc_similarity
            
        if 'nca' in features and 'discriminator' in features:
            nca_disc_similarity = F.cosine_similarity(
                features['nca'],
                features['discriminator'],
                dim=1
            )
            cross_signals['nca_from_disc'] = nca_disc_similarity
            
        if 'transformer' in features:
            if 'generator' in features:
                gen_trans_similarity = F.cosine_similarity(
                    features['generator'],
                    features['transformer'],
                    dim=1
                )
                cross_signals['transformer_from_generator'] = gen_trans_similarity
                
            if 'nca' in features:
                nca_trans_similarity = F.cosine_similarity(
                    features['nca'],
                    features['transformer'],
                    dim=1
                )
                cross_signals['transformer_from_nca'] = nca_trans_similarity
        
        return cross_signals

    def forward(self, images_dict, epoch_progress, batch_progress, target_performance=None):
        # Extract cross-model features with enhanced structure awareness
        features = self.extract_cross_features(images_dict)
        
        # Compute cross-learning signals
        cross_signals = self.compute_cross_learning_signals(features)
        
        # Get structural importance weights
        structural_weights = torch.ones(self.num_models, device=next(self.parameters()).device)
        
        # Weight the signals for each model
        gen_signals = []
        nca_signals = []
        
        if 'generator' in features:
            gen_signals.append(features['generator'].mean(dim=1))
        if 'nca' in features:
            nca_signals.append(features['nca'].mean(dim=1))
            
        gen_signals = torch.stack(gen_signals) if gen_signals else torch.zeros(1, device=next(self.parameters()).device)
        nca_signals = torch.stack(nca_signals) if nca_signals else torch.zeros(1, device=next(self.parameters()).device)
        
        # Apply structural weights
        gen_signals = gen_signals * structural_weights.view(-1, 1)
        nca_signals = nca_signals * structural_weights.view(-1, 1)
        
        # Enhanced ensemble prediction with structural emphasis
        all_signals = torch.cat([
            gen_signals.mean(dim=0, keepdim=True),
            nca_signals.mean(dim=0, keepdim=True),
            cross_signals.get('transformer_from_discriminator', torch.zeros_like(gen_signals[0])).unsqueeze(0),
            torch.ones_like(gen_signals[0]).unsqueeze(0) * 0.5,
            torch.ones_like(gen_signals[0]).unsqueeze(0) * 0.5
        ], dim=0)
        
        ensemble_prediction = self.ensemble_predictor(all_signals.t())
        
        return {
            'weighted_gen_quality': gen_signals.mean(dim=0),
            'weighted_nca_quality': nca_signals.mean(dim=0),
            'ensemble_prediction': ensemble_prediction.squeeze(),
            'gen_weights': structural_weights,
            'nca_weights': structural_weights,
            'cross_signals': cross_signals
        } 