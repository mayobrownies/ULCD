import torch
import torch.nn as nn
import torch.optim as optim
from . import config
import numpy as np
import copy
from collections import defaultdict
from . import config

class FedPAGRClient:
    def __init__(self, model, train_loader, client_id, num_classes, device):
        self.model = model
        self.train_loader = train_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.device = device
        
        # Hardcoded params as requested
        self.beta = 0.1 # Global reg weight
        self.lambda_anchor = 0.0 # Anchor weight (using hard anchoring instead implies this might be 0, or if soft, >0)
        # "5.1 Classifier anchoring... W = P ... or soft constraint"
        # I will implement HARD anchoring at start of round, and maybe soft constraint during training?
        # User said: "W_k,c <- P_c OR soft constraint".
        # I will do HARD anchoring at start.
        
        self.current_prototypes = None

    def train(self, epochs, server_prototypes, round_num=0):
        self.model.train()
        self.current_prototypes = server_prototypes
        
        # 5.1 Classifier Anchoring (Hard)
        # Ensure classifier weights match global prototypes
        # The model classifier is linear: Wx + b. We set W to P.
        # Check model structure.
        # models.py: self.classifier = nn.Linear(feature_dim, num_classes)
        # P is [D]. W is [C, D].
        if server_prototypes is not None:
             with torch.no_grad():
                 for c in range(self.num_classes):
                     if c in server_prototypes:
                         # Normalize prototype
                         p = torch.nn.functional.normalize(server_prototypes[c], p=2, dim=0)
                         # Set row c of classifier weight
                         self.model.classifier.weight.data[c] = p
                         # Set bias to 0 for cosine-like behavior or keep as is?
                         # Usually 0 for strict proto matching.
                         if self.model.classifier.bias is not None:
                             self.model.classifier.bias.data[c].fill_(0)
        
        optimizer = optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                # Forward
                # We need features for L_global
                # models.py forward returns logits.
                # But we can use `get_consensus_features` or `forward(..., return_protos=True)`
                # models.py `forward(x, return_consensus=True)` returns (logits, consensus_features)
                # FedPAGR uses "consensus features" (projected).
                
                logits, features = self.model(x, return_consensus=True)
                # features are normalized in models.py if return_consensus=True?
                # Let's check models.py:
                # `consensus = consensus / torch.norm(consensus, dim=1, keepdim=True).clamp(min=1e-6)` -> Yes.
                
                loss_ce = criterion(logits, y)
                
                loss_global = 0.0
                # (b) Global Regularization Loss (L_global)
                if server_prototypes is not None: 
                    protos = []
                    # Ensure order is 0..C-1
                    # server_prototypes is Dict[c, Tensor]
                    for c in range(self.num_classes):
                        if c in server_prototypes:
                             protos.append(server_prototypes[c].to(self.device))
                        else:
                             # Should not happen ideally if initialized fully
                             # Or handle gracefully? For now assume all exist.
                             pass
                    
                    if len(protos) == self.num_classes:
                        W_global = torch.stack(protos) # [C, D]
                        # Cosine similarity -> We need normalized features
                        # consensus_features are already normalized in model.forward(return_consensus=True)
                        
                        # logits_global = sim(f, W_global)
                        # but we need temperature scaling (beta)
                        
                        # Simply: logits = (f @ W_global.T) / beta
                        # f: [B, D], W_global: [C, D]
                        
                        beta = config.FEDPAGR_BETA 
                        
                        logits_global = torch.matmul(features, W_global.T) / beta
                        
                        loss_global = criterion(logits_global, y)
                        
                # 5.3 Final Objective
                # L = L_CE + lambda_g * L_global + lambda_a * L_anchor
                # We assume lambda_g is implicit 1.0 logic in the formula or tunable.
                # User says: "L_k = L_CE ... + lambda_g L_global".
                # I'll use lambda_g = 1.0.
                
                # (c) Uniform Entropy Regularization (L_entropy)
                # Encourage features to utilize the entire prototype space
                # L_ent = -1/C * Sum_c log( exp(s_c) / Sum_j exp(s_j) )
                # This seems like maximizing entropy of the softmax distribution over prototypes.
                # Maximizing entropy = Minimizing negative entropy.
                # Formula provided: - 1/C * Sum log (softmax_c)
                # Wait. Standard CE is -log(p_y).
                # Entropy H(p) = -Sum p log p.
                # User's formula: -1/C Sum_c log(p_c). This is minimizing CrossEntropy between Uniform and p?
                # KL(Uniform || p) = Sum (1/C) log((1/C)/p) = const - 1/C Sum log p.
                # Minimizing KL(Uniform || p) => Maximizing Sum log p.
                # So we want to MINIMIZE: - 1/C Sum log p_c.
                # This encourages p to be uniform.
                
                loss_entropy = 0.0
                if server_prototypes is not None and len(server_prototypes) == self.num_classes:
                     # Re-use logits_global from above if available, or recompute
                     # logits_global is [B, C].
                     # p = softmax(logits_global)
                     log_probs = torch.nn.functional.log_softmax(logits_global, dim=1) # [B, C]
                     # We want mean over batch, and mean over classes?
                     # User formula: -1/C Sum_c ... (for a single sample?)
                     # We averaging over batch.
                     # L_ent = - mean_batch ( mean_class ( log_probs ) )
                     loss_entropy = - torch.mean( torch.mean(log_probs, dim=1) )
                     
                total_loss = loss_ce + loss_global + (config.FEDPAGR_ENTROPY_WEIGHT * loss_entropy)
                
                total_loss.backward()
                optimizer.step()

    def compute_prototypes(self):
        """
        Compute local class prototypes: Mean of normalized features.
        """
        self.model.eval()
        sum_features = defaultdict(list)
        
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, features = self.model(x, return_consensus=True) # Normalized features
                
                for i in range(len(y)):
                    label = y[i].item()
                    feat = features[i].detach() # [D]
                    sum_features[label].append(feat)
                    
        # Average and Normalize
        client_prototypes = {}
        for c, feats in sum_features.items():
            if feats:
                stack = torch.stack(feats)
                mean_feat = stack.mean(dim=0)
                # Normalize
                proto = torch.nn.functional.normalize(mean_feat, p=2, dim=0)
                client_prototypes[c] = proto.cpu() # Send to CPU for transport
                
        return client_prototypes
