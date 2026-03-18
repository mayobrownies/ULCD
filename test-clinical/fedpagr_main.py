import torch
import numpy as np
import random
import os
import time
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .models import ClinicalBERTFrozen, ClinicalBERTFT2, ClinicalBERTFT4
from .fedpagr_client import FedPAGRClient
from .fedpagr_server import FedPAGRServer
from .logger import FLLogger
from .data_utils import get_heterogeneous_dataloaders
from .multi_gpu_utils import parallel_train_clients
from . import config

# Parses command-line arguments for the Federated Learning script.
def parse_args():
    parser = argparse.ArgumentParser(description='FedPAGR Federated Learning (Clinical)')
    parser.add_argument('--num_rounds', type=int, default=config.NUM_ROUNDS)
    parser.add_argument('--num_clients', type=int, default=config.NUM_CLIENTS)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default="Clinical_FedPAGR")
    return parser.parse_args()

# Sets random seeds for reproducibility across numpy, torch, and random modules.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Evaluates a single model on a given test loader and returns metrics.
def evaluate_model(model, test_loader, device, num_classes=6):
    original_device = next(model.parameters()).device
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['label'] 
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            
    model.to(original_device)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == c).mean()
        else:
            per_class_acc[c] = 0.0

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'per_class_accuracy': per_class_acc
    }

# Evaluates an ensemble of models by averaging their softmax probabilities.
def evaluate_ensemble(models, test_loader, device):
    all_probs = []
    all_labels = []
    
    for m in models: m.cpu()
        
    with torch.no_grad():
        for batch in test_loader:
             input_ids = batch['input_ids'].to(device)
             attention_mask = batch['attention_mask'].to(device)
             y = batch['label']
             all_labels.extend(y.numpy())
             
             batch_probs_sum = None
             
             for model in models:
                 model.to(device)
                 model.eval()
                 logits = model(input_ids, attention_mask=attention_mask)
                 probs = torch.nn.functional.softmax(logits, dim=1)
                 
                 if batch_probs_sum is None:
                     batch_probs_sum = probs
                 else:
                     batch_probs_sum += probs
                 
                 model.cpu() 
                 
             avg_probs = batch_probs_sum / len(models)
             preds = avg_probs.argmax(dim=1).cpu().numpy()
             all_probs.extend(preds)

    all_preds = np.array(all_probs)
    all_labels = np.array(all_labels)


    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }

# Evaluates each client model on its own local test set and aggregates results.
def evaluate_all_clients_local(models, client_test_loaders, device, global_prototypes=None, num_classes=6, client_classes=None):
    total_correct = 0
    total_samples = 0
    client_accs = []

    for client_id, (model, test_loader) in enumerate(zip(models, client_test_loaders)):
        model = model.to(device)
        model.eval()
        correct = 0
        samples = 0
        
        if test_loader is None or len(test_loader) == 0:
            client_accs.append(0.0)
            model.cpu() 
            continue

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                y = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                samples += y.size(0)

        client_acc = correct / samples if samples > 0 else 0
        client_accs.append(client_acc)
        total_correct += correct
        total_samples += samples
        
        model.cpu() 

    weighted_acc = total_correct / total_samples if total_samples > 0 else 0
    mean_acc = np.mean(client_accs)

    return {
        'weighted_accuracy': weighted_acc,
        'mean_accuracy': mean_acc,
        'client_accuracies': client_accs
    }

# Evaluates a model using prototype-based classification with cosine similarity.
def evaluate_model_with_prototypes(model, test_loader, device, global_prototypes, num_classes=6):
    original_device = next(model.parameters()).device
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    valid_keys = [c for c in range(num_classes) if c in global_prototypes]
    if len(valid_keys) < num_classes:
        res = evaluate_model(model, test_loader, device, num_classes)
        model.to(original_device)
        return res
        
    proto_tensor = torch.stack([global_prototypes[c].to(device) for c in range(num_classes)])

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['label']

            consensus_feats = model.get_consensus_features(input_ids, attention_mask=attention_mask)
            sims = torch.nn.functional.cosine_similarity(
                consensus_feats.unsqueeze(1),
                proto_tensor.unsqueeze(0),
                dim=2
            )
            preds = sims.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            
    model.to(original_device)


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == c).mean()
        else:
            per_class_acc[c] = 0.0
            
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'per_class_accuracy': per_class_acc
    }

# Evaluates an ensemble using averaged prototype similarities.
def evaluate_ensemble_with_prototypes(models, test_loader, device, global_prototypes, num_classes=6):
    proto_tensor = torch.stack([global_prototypes[c].to(device) for c in range(num_classes)])

    all_preds = []
    all_labels = []
    
    for m in models: m.cpu()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['label']
            all_labels.extend(y.numpy())

            batch_sims_sum = None
            
            for model in models:
                 model.to(device)
                 model.eval()
                 consensus_feats = model.get_consensus_features(input_ids, attention_mask=attention_mask)
                 sims = torch.nn.functional.cosine_similarity(
                     consensus_feats.unsqueeze(1),
                     proto_tensor.unsqueeze(0),
                     dim=2
                 )
                 
                 if batch_sims_sum is None:
                     batch_sims_sum = sims
                 else:
                     batch_sims_sum += sims
                 
                 model.cpu() 

            avg_sims = batch_sims_sum / len(models)
            preds = avg_sims.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)


    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }

# Saves the current state of the server and client models to a checkpoint file.
def save_checkpoint(clients, server, round_num, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'round': round_num,
        'prototypes': server.global_prototypes,
        'models': {i: c.model.state_dict() for i, c in enumerate(clients)}
    }
    torch.save(checkpoint, f"{output_dir}/checkpoint_r{round_num}.pt")
    print(f"Saved at round {round_num}")

# Main execution function that orchestrates the Federated Learning process.
def run():
    args = parse_args()
    if args.gpu is not None:
        if ',' in args.gpu:
            config.GPU_IDS = [int(x) for x in args.gpu.split(',')]
        else:
            config.GPU_IDS = [int(args.gpu)]
    if args.num_rounds is not None:
        config.NUM_ROUNDS = args.num_rounds
    if args.num_clients is not None:
        config.NUM_CLIENTS = args.num_clients
    
    set_seed(config.SEED)
    
    if config.USE_CUDA and torch.cuda.is_available():
        torch.cuda.set_device(config.GPU_IDS[0])
    device = torch.device(f"cuda:{config.GPU_IDS[0]}" if (torch.cuda.is_available() and config.USE_CUDA) else "cpu")
    
    logger = FLLogger(output_dir=config.OUTPUT_DIR, experiment_name=args.exp_name)
    
    print("="*80)
    print("FedPAGR Implementation (Clinical)")
    print(f"Dataset: {config.DATASET}")
    print(f"Device: {device}")
    print(f"Models: {', '.join(config.MODEL_TYPES)}")
    print(f"Feature Dimension: {config.FEATURE_DIM}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print("="*80)
    
    client_loaders, _, test_loader, client_info = get_heterogeneous_dataloaders()
    num_classes = config.NUM_CLASSES
    client_test_loaders = client_info['client_test_loaders']
    
    model_classes = [ClinicalBERTFrozen, ClinicalBERTFT2, ClinicalBERTFT4]
    model_names = ["ClinicalBERT-Frozen", "ClinicalBERT-FT2", "ClinicalBERT-FT4"]
    models = []
    
    print("\nClient-Model Assignment:")
    for i in range(config.NUM_CLIENTS):
        model_idx = i % len(model_classes)
        model = model_classes[model_idx](num_classes=num_classes, feature_dim=config.FEATURE_DIM).to(device)
        models.append(model)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Client {i}: {model_names[model_idx]} ({params:,} trainable params)")

    clients = [
        FedPAGRClient(model.to(device), cl, i, num_classes=num_classes, device=device)
        for i, (model, cl) in enumerate(zip(models, client_loaders))
    ]

    server = FedPAGRServer(feature_dim=config.FEATURE_DIM, num_classes=num_classes, device=device)
    
    num_join_clients = max(1, int(config.NUM_CLIENTS * config.JOIN_RATIO))

    print(f"\nStarting Training for {config.NUM_ROUNDS} rounds...")

    for round_num in range(1, config.NUM_ROUNDS + 1):
        print(f"\n{'='*80}")
        print(f"Round {round_num}/{config.NUM_ROUNDS}")
        print(f"{'='*80}")
        
        selected_indices = np.random.choice(len(clients), num_join_clients, replace=False)
        selected_clients = [clients[i] for i in selected_indices]
        print(f"Selected clients: {sorted(selected_indices.tolist())}")
        
        round_start_time = time.time()
        global_protos = server.broadcast() 
        
        client_protos_list = []
        if config.USE_MULTI_GPU:
             active_clients = [c for c in selected_clients if c.train_loader is not None]
             
             if active_clients:
                 print(f"Training {len(active_clients)} clients on {len(config.GPU_IDS)} GPUs: {config.GPU_IDS}")
                 trained_clients = parallel_train_clients(active_clients, config.EPOCHS_PER_ROUND, global_protos, round_num)
                 
                 for client in trained_clients:
                     print(f"Server: Computing prototypes for Client {client.client_id} on {device}...")
                     client.model = client.model.to(device)
                     client.device = device
                     
                     c_protos = client.compute_prototypes()

                     client_protos_list.append(c_protos)
                     
                     client.model = client.model.cpu()
                     client.device = torch.device('cpu')
                     if torch.cuda.is_available():
                         torch.cuda.empty_cache()

        else:
            for client in selected_clients:
                if client.train_loader is None:
                    continue 
                print(f"Client {client.client_id} Local training")
                client.train(epochs=config.EPOCHS_PER_ROUND, server_prototypes=global_protos, round_num=round_num)
                c_protos = client.compute_prototypes()
                client_protos_list.append(c_protos)
            
        print(f"Server: Collecting client prototypes")
        if client_protos_list:
            server.aggregate_prototypes(client_protos_list)
            server.refine_prototypes()
        else:
            print("No prototypes collected this round.")
        
        round_end_time = time.time()
        logger.log_training_time(round_num, round_end_time - round_start_time)

        logger.log_round(round_num)
        
        if round_num % config.EVAL_FREQ == 0 or round_num == config.NUM_ROUNDS:
            print(f"\nEvaluation Round {round_num}")
            
            local_eval = evaluate_all_clients_local([c.model for c in clients], client_test_loaders, device, global_protos, num_classes, None)
            print(f"Local Test: Weighted={local_eval['weighted_accuracy']:.4f}, Mean={local_eval['mean_accuracy']:.4f}")
            
            clients_metrics = []
            if config.ENABLE_CLIENT_GLOBAL_EVAL:
                print("Evaluating all clients on global test set...")
                for i, client in enumerate(clients):
                     if global_protos and len(global_protos) == num_classes:
                         metrics = evaluate_model_with_prototypes(client.model, test_loader, device, global_protos, num_classes)
                     else:
                         metrics = evaluate_model(client.model, test_loader, device, num_classes)
                     clients_metrics.append(metrics)
            else:
                pass


            if config.ENABLE_ENSEMBLE_EVAL:
                if global_protos and len(global_protos) == num_classes:
                    ensemble_metrics = evaluate_ensemble_with_prototypes([c.model for c in clients], test_loader, device, global_protos, num_classes)
                else:
                    ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
                print(f"Global Test Ensemble: Acc={ensemble_metrics['accuracy']:.4f}")
            else:
                 ensemble_metrics = {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 'precision': 0.0, 'recall': 0.0}

            
            logger.log_evaluation(round_num, clients_metrics, ensemble_metrics, local_metrics=local_eval)
            
            current_acc = local_eval['weighted_accuracy']
                
        if round_num % 10 == 0 or True: 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Calculate Communication Cost (Prototypes: C x D x 4 bytes)
    upload_bytes = config.NUM_CLASSES * config.FEATURE_DIM * 4
    download_bytes = config.NUM_CLASSES * config.FEATURE_DIM * 4
    bytes_per_round_per_client = upload_bytes + download_bytes
    total_bytes = bytes_per_round_per_client * config.NUM_CLIENTS * config.NUM_ROUNDS
    total_mb = total_bytes / (1024 * 1024)
    efficiency = 0 / total_mb if total_mb > 0 else 0

    print(f"Bytes per round per client: {bytes_per_round_per_client:,.0f} bytes ({bytes_per_round_per_client/1024:.2f} KB)")
    print(f"Total communication:        {total_bytes:,.0f} bytes ({total_mb:.2f} MB)")
    print(f"Communication efficiency:   {efficiency:.4f} (Accuracy/MB)")

    logger.log_communication(bytes_per_round_per_client, total_bytes, total_mb, efficiency)

    logger.save_results()
    print("Completed")

if __name__ == "__main__":
    run()
