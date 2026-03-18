import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from collections import OrderedDict
from . import config

# Initializes the Federated Learning server with global prototypes and parameters.
class FedPAGRServer:
    def __init__(self, feature_dim, num_classes, device):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        
        self.global_prototypes = {}
        for c in range(num_classes):
             p = torch.randn(feature_dim, device=device)
             self.global_prototypes[c] = torch.nn.functional.normalize(p, p=2, dim=0)
             
        self.prev_global_prototypes = copy.deepcopy(self.global_prototypes)

    # Aggregates client prototypes into global consensus prototypes using weighted averaging.
    def aggregate_prototypes(self, client_prototypes_list, weights=None):
        if weights is None:
            n_clients = len(client_prototypes_list)
            weights = [1.0 / n_clients] * n_clients
            
        class_prototypes = {c: [] for c in range(self.num_classes)}
        class_weights = {c: [] for c in range(self.num_classes)}
        
        for idx, client_protos in enumerate(client_prototypes_list):
            w = weights[idx]
            for c, p in client_protos.items():
                class_prototypes[c].append(p.to(self.device))
                class_weights[c].append(w)
                
        new_prototypes = {}
        
        for c in range(self.num_classes):
            if not class_prototypes[c]:
                new_prototypes[c] = self.global_prototypes[c].clone()
                continue
                
            protos_c = torch.stack(class_prototypes[c]) 
            weights_c = torch.tensor(class_weights[c], device=self.device).view(-1, 1) 
            
            weighted_sum = (protos_c * weights_c).sum(dim=0)
            
            new_prototypes[c] = torch.nn.functional.normalize(weighted_sum, p=2, dim=0)
            
        self.global_prototypes = new_prototypes
        
        self.class_prototypes_cache = class_prototypes 

    # Refines global prototypes using geometric regularization losses (agreement, separation, temporal).
    def refine_prototypes(self):
        LAMBDA_S = config.FEDPAGR_LAMBDA_S
        LAMBDA_T = config.FEDPAGR_LAMBDA_T
        MARGIN = config.FEDPAGR_MARGIN
        LR = config.FEDPAGR_REFINE_LR
        STEPS = config.FEDPAGR_REFINE_STEPS
        
        protos_list = [self.global_prototypes[c].detach().clone() for c in range(self.num_classes)]
        protos_tensor = torch.stack(protos_list).requires_grad_(True) 
        
        optimizer = optim.SGD([protos_tensor], lr=LR, momentum=0.9)
        
        prev_protos_tensor = torch.stack([self.prev_global_prototypes[c] for c in range(self.num_classes)]).detach()
        
        for step in range(STEPS):
            optimizer.zero_grad()
            
            current_protos_norm = torch.nn.functional.normalize(protos_tensor, p=2, dim=1)
            
            loss = 0.0
            
            l_agree = 0.0
            for c in range(self.num_classes):
                if c in self.class_prototypes_cache and len(self.class_prototypes_cache[c]) > 0:
                    client_protos_c = torch.stack(self.class_prototypes_cache[c]).detach() 
                    
                    sims = torch.matmul(client_protos_c, current_protos_norm[c].unsqueeze(1)).squeeze() 
                    l_agree += (1.0 - sims).sum()
            
            loss += l_agree
            
            similarity_matrix = torch.matmul(current_protos_norm, current_protos_norm.T)
            
            mask = torch.eye(self.num_classes, device=self.device).bool()
            off_diagonal_sims = similarity_matrix[~mask]
            
            l_sep = torch.clamp(off_diagonal_sims - MARGIN, min=0).sum()

            
            l_temp = torch.nn.functional.mse_loss(current_protos_norm, prev_protos_tensor, reduction='sum')
            
            total_loss = l_agree + (LAMBDA_S * l_sep) + (LAMBDA_T * l_temp)
            
            total_loss.backward()
            optimizer.step()
            
        with torch.no_grad():
             for c in range(self.num_classes):
                 self.global_prototypes[c] = torch.nn.functional.normalize(protos_tensor[c], p=2, dim=0)

    # Updates previous prototypes and broadcasts the current global prototypes to clients.
    def broadcast(self):
        self.prev_global_prototypes = copy.deepcopy(self.global_prototypes)
        return self.global_prototypes
