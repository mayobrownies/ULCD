import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CNNModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
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

    def forward(self, x, return_protos=False, use_consensus=True):
        x = self.features(x)
        protos = self.projection(x)

        if use_consensus:
            consensus = self.consensus_projection(protos)
            consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)
            logits = self.classifier(consensus)
        else:
            logits = self.classifier(protos)

        if return_protos:
            log_probs = F.log_softmax(logits, dim=1)
            return log_probs, protos
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
    def __init__(self, num_classes=10, feature_dim=512):
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
        self.projection = nn.Linear(512, feature_dim)
        self.consensus_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False, use_consensus=True):
        x = self.features(x)
        protos = self.projection(x)

        if use_consensus:
            consensus = self.consensus_projection(protos)
            consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)
            logits = self.classifier(consensus)
        else:
            logits = self.classifier(protos)

        if return_protos:
            log_probs = F.log_softmax(logits, dim=1)
            return log_probs, protos
        return logits

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)

    def get_consensus_features(self, x):
        latent = self.get_features(x)
        consensus_latent = self.consensus_projection(latent)
        norm = torch.norm(consensus_latent, dim=1, keepdim=True).clamp(min=1e-6)
        return consensus_latent / norm

class ResNetModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        base = resnet18(pretrained=False)
        base.fc = nn.Identity()
        self.features = base
        self.projection = nn.Linear(512, feature_dim)
        self.consensus_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False, use_consensus=True):
        feat = self.features(x)
        protos = self.projection(feat)

        if use_consensus:
            consensus = self.consensus_projection(protos)
            consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)
            logits = self.classifier(consensus)
        else:
            logits = self.classifier(protos)

        if return_protos:
            log_probs = F.log_softmax(logits, dim=1)
            return log_probs, protos
        return logits

    def get_features(self, x):
        feat = self.features(x)
        return self.projection(feat)

    def get_consensus_features(self, x):
        latent = self.get_features(x)
        consensus_latent = self.consensus_projection(latent)
        norm = torch.norm(consensus_latent, dim=1, keepdim=True).clamp(min=1e-6)
        return consensus_latent / norm
