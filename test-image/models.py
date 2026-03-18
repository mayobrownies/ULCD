import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, mobilenet_v2, googlenet
from . import config

class CNNModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=config.FEATURE_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.projection = nn.Linear(64 * 8 * 8, feature_dim)
        self.consensus_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False, return_consensus=False, use_consensus=True):
        x = self.features(x)
        protos = self.projection(x)

        if use_consensus or return_consensus:
            consensus = self.consensus_projection(protos)
            consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)
            logits = self.classifier(consensus)
        else:
            consensus = None
            protos = F.relu(protos)
            logits = self.classifier(protos)

        if return_protos:
            return logits, protos
        if return_consensus:
            return logits, consensus
        return logits

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)

    def get_consensus_features(self, x):
        latent = self.get_features(x)
        consensus_latent = self.consensus_projection(latent)
        norm = torch.norm(consensus_latent, dim=1, keepdim=True).clamp(min=1e-6)
        return consensus_latent / norm


class MLPModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=config.FEATURE_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        # MLP output is 512. Feature Dim is 512.
        self.projection = nn.Identity()
        self.consensus_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False, return_consensus=False, use_consensus=True):
        x = self.features(x)
        protos = self.projection(x)

        if use_consensus or return_consensus:
            consensus = self.consensus_projection(protos)
            consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)
            logits = self.classifier(consensus)
        else:
            consensus = None
            protos = F.relu(protos)
            logits = self.classifier(protos)

        if return_protos:
            return logits, protos
        if return_consensus:
            return logits, consensus
        return logits

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)

    def get_consensus_features(self, x):
        latent = self.get_features(x)
        consensus_latent = self.consensus_projection(latent)
        norm = torch.norm(consensus_latent, dim=1, keepdim=True).clamp(min=1e-6)
        return consensus_latent / norm


class ResNet18Model(nn.Module):
    def __init__(self, num_classes=10, feature_dim=config.FEATURE_DIM):
        super().__init__()
        base = resnet18(pretrained=False)
        base.fc = nn.Identity()
        self.features = base
        # Output of ResNet18 is 512. Feature Dim is 512.
        # Reference implementation uses AdaptiveAvgPool1d(512), which is identity-like for (N, 512).
        # We use Identity to avoid extra learnable parameters that might rotate the feature space.
        self.projection = nn.Identity()
        
        self.consensus_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False, return_consensus=False, use_consensus=True):
        x = self.features(x)
        protos = self.projection(x)

        if use_consensus or return_consensus:
            consensus = self.consensus_projection(protos)
            consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)
            logits = self.classifier(consensus)
        else:
            consensus = None
            logits = self.classifier(protos)

        if return_protos:
            return logits, protos
        if return_consensus:
            return logits, consensus
        return logits

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)

    def get_consensus_features(self, x):
        latent = self.get_features(x)
        consensus_latent = self.consensus_projection(latent)
        norm = torch.norm(consensus_latent, dim=1, keepdim=True).clamp(min=1e-6)
        return consensus_latent / norm


class GoogLeNetModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=config.FEATURE_DIM):
        super().__init__()
        base = googlenet(pretrained=False, aux_logits=False, init_weights=True)
        base.fc = nn.Identity()
        self.features = base
        # GoogLeNet output is 1024
        self.projection = nn.Linear(1024, feature_dim)
        self.consensus_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False, return_consensus=False, use_consensus=True):
        x = self.features(x)
        protos = self.projection(x)

        if use_consensus or return_consensus:
            consensus = self.consensus_projection(protos)
            consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)
            logits = self.classifier(consensus)
        else:
            consensus = None
            logits = self.classifier(protos)

        if return_protos:
            return logits, protos
        if return_consensus:
            return logits, consensus
        return logits

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)

    def get_consensus_features(self, x):
        latent = self.get_features(x)
        consensus_latent = self.consensus_projection(latent)
        norm = torch.norm(consensus_latent, dim=1, keepdim=True).clamp(min=1e-6)
        return consensus_latent / norm


class MobileNetModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=config.FEATURE_DIM):
        super().__init__()
        base = mobilenet_v2(pretrained=False)
        base.classifier = nn.Identity()
        self.features = base
        self.projection = nn.Linear(1280, feature_dim)
        
        self.consensus_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False, return_consensus=False, use_consensus=True):
        x = self.features(x)
        protos = self.projection(x)

        if use_consensus or return_consensus:
            consensus = self.consensus_projection(protos)
            consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)
            logits = self.classifier(consensus)
        else:
            consensus = None
            logits = self.classifier(protos)

        if return_protos:
            return logits, protos
        if return_consensus:
            return logits, consensus
        return logits

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)

    def get_consensus_features(self, x):
        latent = self.get_features(x)
        consensus_latent = self.consensus_projection(latent)
        norm = torch.norm(consensus_latent, dim=1, keepdim=True).clamp(min=1e-6)
        return consensus_latent / norm


ResNetModel = ResNet18Model
