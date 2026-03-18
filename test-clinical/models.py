import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from . import config

class ClinicalBERT(nn.Module):
    def __init__(self, num_classes=6, feature_dim=config.FEATURE_DIM, fine_tune_layers=0):
        super().__init__()
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Freezing logic
        if fine_tune_layers == 0:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            # Freeze all first
            for param in self.bert.parameters():
                param.requires_grad = False
            
            # Unfreeze last N layers of encoder
            total_layers = len(self.bert.encoder.layer)
            start_layer = total_layers - fine_tune_layers
            for i in range(start_layer, total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            
            # Also unfreeze pooler if present
            if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True

        # Projection Head (768 -> feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.ReLU() 
        )

        # Consensus Head (for FedPAGR)
        self.consensus_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, return_protos=False, return_consensus=False, use_consensus=True):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        protos = self.projection(pooled)

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

    def get_features(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        return self.projection(pooled)

    def get_consensus_features(self, input_ids, attention_mask=None):
        latent = self.get_features(input_ids, attention_mask)
        consensus_latent = self.consensus_projection(latent)
        norm = torch.norm(consensus_latent, dim=1, keepdim=True).clamp(min=1e-6)
        return consensus_latent / norm


class ClinicalBERTFrozen(ClinicalBERT):
    def __init__(self, num_classes=6, feature_dim=config.FEATURE_DIM):
        super().__init__(num_classes, feature_dim, fine_tune_layers=0)

class ClinicalBERTFT2(ClinicalBERT):
    def __init__(self, num_classes=6, feature_dim=config.FEATURE_DIM):
        super().__init__(num_classes, feature_dim, fine_tune_layers=2)

class ClinicalBERTFT4(ClinicalBERT):
    def __init__(self, num_classes=6, feature_dim=config.FEATURE_DIM):
        super().__init__(num_classes, feature_dim, fine_tune_layers=4)
