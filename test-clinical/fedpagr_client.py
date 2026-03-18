import torch
import torch.nn as nn
import torch.optim as optim
from . import config
import numpy as np
import copy
from collections import defaultdict
from . import config

# Initializes the Federated Learning client with model, data, and parameters.
class FedPAGRClient:
    def __init__(self, model, train_loader, client_id, num_classes, device):
        self.model = model
        self.train_loader = train_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.device = device
        
        self.beta = config.FEDPAGR_BETA
        self.lambda_anchor = 0.0 
        
        self.current_prototypes = None

    # Performs local training on the client's private data for a specified number of epochs.
    def train(self, epochs, server_prototypes, round_num=0):
        self.model.train()
        self.current_prototypes = server_prototypes
        
        if server_prototypes is not None:
             with torch.no_grad():
                 for c in range(self.num_classes):
                     if c in server_prototypes:
                         p = torch.nn.functional.normalize(server_prototypes[c], p=2, dim=0)
                         self.model.classifier.weight.data[c] = p
                         if self.model.classifier.bias is not None:
                             self.model.classifier.bias.data[c].fill_(0)
        
        optimizer = optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
        
        for epoch in range(epochs):
            print(f"Client {self.client_id} [GPU {self.device}] Epoch {epoch} start")
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                y = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                    logits, features = self.model(input_ids, attention_mask=attention_mask, return_consensus=True)
                    
                    loss_ce = criterion(logits, y)
                    
                    loss_global = 0.0
                    if server_prototypes is not None: 
                        protos = []
                        valid_protos = True
                        for c in range(self.num_classes):
                            if c in server_prototypes:
                                 protos.append(server_prototypes[c].to(self.device))
                            else:
                                 valid_protos = False
                                 break
                        
                        if valid_protos and len(protos) == self.num_classes:
                            W_global = torch.stack(protos) 
                            beta = self.beta
                            logits_global = torch.matmul(features.float(), W_global.T.float()) / beta
                            loss_global = criterion(logits_global, y)
                            
                    loss_entropy = 0.0
                    if server_prototypes is not None and valid_protos and len(protos) == self.num_classes:
                         log_probs = torch.nn.functional.log_softmax(logits_global, dim=1) 
                         loss_entropy = - torch.mean( torch.mean(log_probs, dim=1) )
                         
                    total_loss = loss_ce + loss_global + (config.FEDPAGR_ENTROPY_WEIGHT * loss_entropy)
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if batch_idx % 50 == 0:
                     print(f"Client {self.client_id} [GPU {self.device}] Epoch {epoch} Batch {batch_idx}/{len(self.train_loader)} Loss: {total_loss.item():.4f}")

    # Computes local class prototypes by averaging feature representations of training data.
    def compute_prototypes(self):
        self.model.eval()
        sum_features = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                y = batch['label'].to(self.device)
                
                _, features = self.model(input_ids, attention_mask=attention_mask, return_consensus=True)
                
                for i in range(len(y)):
                    label = y[i].item()
                    feat = features[i].detach() 
                    sum_features[label].append(feat)
                    
        client_prototypes = {}
        for c, feats in sum_features.items():
            if feats:
                stack = torch.stack(feats)
                mean_feat = stack.mean(dim=0)
                proto = torch.nn.functional.normalize(mean_feat, p=2, dim=0)
                client_prototypes[c] = proto.cpu()
                
        return client_prototypes
