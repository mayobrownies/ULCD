import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from collections import OrderedDict
from . import config

class FedPAGRServer:
    def __init__(self, feature_dim, num_classes, device):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        
        # Global prototypes: Dict[class_idx, Tensor(feature_dim)]
        # Initialize randomly and normalize
        self.global_prototypes = {}
        for c in range(num_classes):
             # Random init
             p = torch.randn(feature_dim, device=device)
             self.global_prototypes[c] = torch.nn.functional.normalize(p, p=2, dim=0)
             
        # Previous round prototypes for temporal stability
        self.prev_global_prototypes = copy.deepcopy(self.global_prototypes)

    def aggregate_prototypes(self, client_prototypes_list, weights=None):
        """
        Aggregate client prototypes into a consensus prototype.
        client_prototypes_list: List of Dict[class_idx, prototype_tensor]
        weights: List of weights for each client (default uniform)
        """
        if weights is None:
            n_clients = len(client_prototypes_list)
            weights = [1.0 / n_clients] * n_clients
            
        # Group by class
        class_prototypes = {c: [] for c in range(self.num_classes)}
        class_weights = {c: [] for c in range(self.num_classes)}
        
        for idx, client_protos in enumerate(client_prototypes_list):
            w = weights[idx]
            for c, p in client_protos.items():
                class_prototypes[c].append(p.to(self.device))
                class_weights[c].append(w)
                
        # Compute Consensus (Weighted Average)
        # P_bar_c = Norm(Sum(w_k * P_k,c))
        new_prototypes = {}
        
        # Store for refinement usage (optional, or we just trust the running P_bar)
        # But per the algo, we refine the *Consensus* P_bar.
        # So first we calculate the raw consensus P_bar.
        
        for c in range(self.num_classes):
            if not class_prototypes[c]:
                # If no client reported this class, keep old or random?
                # Ideally keep old global prototype
                new_prototypes[c] = self.global_prototypes[c].clone()
                continue
                
            protos_c = torch.stack(class_prototypes[c]) # [K, D]
            weights_c = torch.tensor(class_weights[c], device=self.device).view(-1, 1) # [K, 1]
            
            # Weighted sum
            weighted_sum = (protos_c * weights_c).sum(dim=0)
            
            # Normalize
            new_prototypes[c] = torch.nn.functional.normalize(weighted_sum, p=2, dim=0)
            
        # Update raw consensus
        self.global_prototypes = new_prototypes
        
        self.class_prototypes_cache = class_prototypes # Store for L_agree calculation

    def refine_prototypes(self):
        """
        Refine the global prototypes using the geometric regularization losses.
        L_final = L_agree + lambda_s * L_sep + lambda_t * L_temp
        """
        # Hyperparameters from config
        LAMBDA_S = config.FEDPAGR_LAMBDA_S
        LAMBDA_T = config.FEDPAGR_LAMBDA_T
        MARGIN = config.FEDPAGR_MARGIN
        LR = config.FEDPAGR_REFINE_LR
        STEPS = config.FEDPAGR_REFINE_STEPS
        
        # Optimize global_prototypes
        # We need them to be leaf tensors with grad
        
        # Convert dict to a learnable tensor parameter
        # Order implies class ID
        protos_list = [self.global_prototypes[c].detach().clone() for c in range(self.num_classes)]
        protos_tensor = torch.stack(protos_list).requires_grad_(True) # [C, D]
        
        optimizer = optim.SGD([protos_tensor], lr=LR, momentum=0.9)
        
        # Cache previous for L_temp
        prev_protos_tensor = torch.stack([self.prev_global_prototypes[c] for c in range(self.num_classes)]).detach()
        
        for step in range(STEPS):
            optimizer.zero_grad()
            
            # Re-normalize for calculation (optimization happens in unconstrained space, but we project/norm effectively)
            # Actually better to optimize on the manifold or just normalize after step. 
            # Standard way: normalize in forward pass of loss.
            
            current_protos_norm = torch.nn.functional.normalize(protos_tensor, p=2, dim=1)
            
            loss = 0.0
            
            # (a) L_agree: Intra-class agreement
            # Sum_c Sum_k (1 - cos(P_bar_c, P_k,c))
            # We need the client prototypes for this round.
            l_agree = 0.0
            for c in range(self.num_classes):
                if c in self.class_prototypes_cache and len(self.class_prototypes_cache[c]) > 0:
                    # Client protos for class c
                    client_protos_c = torch.stack(self.class_prototypes_cache[c]).detach() # [K, D] -> These are already normalized? Yes from client.
                    
                    # Cosine similarity
                    # P_bar is current_protos_norm[c]
                    # sim = (P_bar . P_k)
                    sims = torch.matmul(client_protos_c, current_protos_norm[c].unsqueeze(1)).squeeze() # [K]
                    l_agree += (1.0 - sims).sum()
            
            loss += l_agree
            
            # (b) L_sep: Inter-class separation
            # Enforce orthogonality or minimum angular distance.
            # L_sep = Sum_{c != j} max(0, cos(P_c, P_j) - m)
            # We want cosine similarity <= m.
            # If sim > m, we penalize (sim - m).
            
            similarity_matrix = torch.matmul(current_protos_norm, current_protos_norm.T)
            
            # Mask diagonal (self-similarity is 1, ignore)
            mask = torch.eye(self.num_classes, device=self.device).bool()
            off_diagonal_sims = similarity_matrix[~mask]
            
            # Penalize if similarity > MARGIN
            l_sep = torch.clamp(off_diagonal_sims - MARGIN, min=0).sum()

            
            # (c) L_temp: Temporal stability
            # ||P(t) - P(t-1)||^2
            # Since P are normalized, this is 2(1 - cos). Equivalent to maximize cos(P_t, P_{t-1}).
            # But we can just use MSE on the vectors.
            l_temp = torch.nn.functional.mse_loss(current_protos_norm, prev_protos_tensor, reduction='sum')
            
            total_loss = l_agree + (LAMBDA_S * l_sep) + (LAMBDA_T * l_temp)
            
            total_loss.backward()
            optimizer.step()
            
        # Update global prototypes
        with torch.no_grad():
             for c in range(self.num_classes):
                 self.global_prototypes[c] = torch.nn.functional.normalize(protos_tensor[c], p=2, dim=0)

    def broadcast(self):
        # Update prev for next round
        self.prev_global_prototypes = copy.deepcopy(self.global_prototypes)
        return self.global_prototypes
