import torch
import numpy as np
import random
import os
import argparse
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .models import CNNModel, MLPModel, ResNet18Model, GoogLeNetModel, MobileNetModel
from .fedpagr_client import FedPAGRClient
from .fedpagr_server import FedPAGRServer
from .logger import FLLogger
from .data_utils import get_heterogeneous_dataloaders
from .multi_gpu_utils import parallel_train_clients
from . import config

def parse_args():
    parser = argparse.ArgumentParser(description='FedPAGR Federated Learning')
    parser.add_argument('--num_rounds', type=int, default=config.NUM_ROUNDS)
    parser.add_argument('--num_clients', type=int, default=config.NUM_CLIENTS)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default="FedPAGR_v2")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, test_loader, device, num_classes=10):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            # y = y.to(device) # We don't actually need y on device for basic metric accumulation if we just cpu() it back.
            # But usually we do model(x) -> device.
            
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy()) # y is likely CPU from loader? Or did we move it?
            # data_utils loaders return tensors. DataLoader yields CPU tensors usually.
            # If we didn't move y to device, .numpy() works.
            # Only if we did y = y.to(device) would we need .cpu(). 
            # In previous fedpagr_main: "x, y = x.to(device), y.to(device)".
            # So y was on device. Thus y.numpy() failed.
            
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

def evaluate_ensemble(models, test_loader, device):
    for model in models:
        model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            ensemble_probs = []
            for model in models:
                logits = model(x)
                probs = torch.nn.functional.softmax(logits, dim=1)
                ensemble_probs.append(probs)

            avg_probs = torch.stack(ensemble_probs).mean(dim=0)
            preds = avg_probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }

def evaluate_all_clients_local(models, client_test_loaders, device, global_prototypes=None, num_classes=10, client_classes=None):
    # Added signature match to fedpagr_main.py (global_prototypes etc) to be safe, though unused maybe?
    # Actually new FedPAGR client uses global regularization but evaluation...
    # Let's match fedpagr_main.py's `evaluate_all_clients_local` exactly.
    
    total_correct = 0
    total_samples = 0
    client_accs = []

    for client_id, (model, test_loader) in enumerate(zip(models, client_test_loaders)):
        model.eval()
        correct = 0
        samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                samples += y.size(0)

        client_acc = correct / samples if samples > 0 else 0
        client_accs.append(client_acc)
        total_correct += correct
        total_samples += samples

    weighted_acc = total_correct / total_samples if total_samples > 0 else 0
    mean_acc = np.mean(client_accs)

    return {
        'weighted_accuracy': weighted_acc,
        'mean_accuracy': mean_acc,
        'client_accuracies': client_accs
    }

def evaluate_model_with_prototypes(model, test_loader, device, global_prototypes, num_classes=10):
    model.eval()
    all_preds = []
    all_labels = []
    
    proto_tensor = torch.stack([global_prototypes[c].to(device) for c in range(num_classes)])

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            # Consensus features check
            consensus_feats = model.get_consensus_features(x)
            sims = torch.nn.functional.cosine_similarity(
                consensus_feats.unsqueeze(1),
                proto_tensor.unsqueeze(0),
                dim=2
            )
            preds = sims.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

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

def evaluate_ensemble_with_prototypes(models, test_loader, device, global_prototypes, num_classes=10):
    for model in models:
        model.eval()
        
    proto_tensor = torch.stack([global_prototypes[c].to(device) for c in range(num_classes)])

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            ensemble_sims = []
            for model in models:
                 consensus_feats = model.get_consensus_features(x)
                 sims = torch.nn.functional.cosine_similarity(
                     consensus_feats.unsqueeze(1),
                     proto_tensor.unsqueeze(0),
                     dim=2
                 )
                 ensemble_sims.append(sims)

            avg_sims = torch.stack(ensemble_sims).mean(dim=0)
            preds = avg_sims.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    
def save_checkpoint(clients, server, round_num, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'round': round_num,
        'prototypes': server.global_prototypes,
        'models': {i: c.model.state_dict() for i, c in enumerate(clients)}
    }
    torch.save(checkpoint, f"{output_dir}/checkpoint_r{round_num}.pt")
    print(f"Saved at round {round_num}")


def run():
    args = parse_args()
    
    # Apply basic config overrides
    if args.gpu is not None:
        if ',' in args.gpu:
            config.GPU_IDS = [int(x) for x in args.gpu.split(',')]
        else:
            config.GPU_IDS = [int(args.gpu)]
    config.NUM_ROUNDS = args.num_rounds
    config.NUM_CLIENTS = args.num_clients
    # config.DATASET and config.PUBLIC_SIZE are now correctly set in config.py
    
    set_seed(config.SEED)
    
    if config.USE_CUDA and torch.cuda.is_available():
        torch.cuda.set_device(config.GPU_IDS[0])
    device = torch.device(f"cuda:{config.GPU_IDS[0]}" if (torch.cuda.is_available() and config.USE_CUDA) else "cpu")
    
    logger = FLLogger(output_dir=config.OUTPUT_DIR, experiment_name=args.exp_name)
    
    print("="*80)
    print("FedPAGR Implementation")
    print(f"Dataset: {config.DATASET}")
    print(f"Device: {device}")
    print(f"Models: {', '.join(config.MODEL_TYPES)}")
    print(f"Feature Dimension: {config.FEATURE_DIM}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print(f"Server Refinement: Enabled (Params from config)")
    print("="*80)
    
    # Data
    client_loaders, _, test_loader, client_info = get_heterogeneous_dataloaders()
    num_classes = client_info['num_classes']
    client_test_loaders = client_info['client_test_loaders']
    logger.set_client_classes(client_info['classes'])
    
    # Models & Clients
    model_classes = [CNNModel, MLPModel, ResNet18Model, GoogLeNetModel, MobileNetModel]
    model_names = ["CNN", "MLP", "ResNet18", "GoogLeNet", "MobileNet"]
    models = []
    
    print("\nClient-Model Assignment:")
    for i in range(config.NUM_CLIENTS):
        model_idx = i % len(model_classes)
        # Force feature_dim from config (512)
        model = model_classes[model_idx](num_classes=num_classes, feature_dim=config.FEATURE_DIM).to(device)
        models.append(model)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Client {i}: {model_names[model_idx]} ({params:,} params)")

    clients = [
        FedPAGRClient(model.to(device), cl, i, num_classes=num_classes, device=device)
        for i, (model, cl) in enumerate(zip(models, client_loaders))
    ]

    # Server
    # Note: New server has NO MODEL initialization as per prompt "no model, no data"
    server = FedPAGRServer(feature_dim=config.FEATURE_DIM, num_classes=num_classes, device=device)
    
    num_join_clients = int(config.NUM_CLIENTS * config.JOIN_RATIO)

    print(f"\nStarting Training for {config.NUM_ROUNDS} rounds...")

    for round_num in range(1, config.NUM_ROUNDS + 1):
        print(f"\n{'='*80}")
        print(f"Round {round_num}/{config.NUM_ROUNDS}")
        print(f"{'='*80}")
        
        # 1. Select Clients
        selected_indices = np.random.choice(len(clients), num_join_clients, replace=False)
        selected_clients = [clients[i] for i in selected_indices]
        print(f"Selected clients: {sorted(selected_indices.tolist())}")
        
        round_start_time = time.time()
        
        # 2. Broadcast Global Prototypes
        # server.broadcast() updates prev_protos internally + returns current
        global_protos = server.broadcast() 
        # Note: global_protos is a Dict[c, Tensor]
        
        # 3. Client Training
        client_protos_list = []
        for client in selected_clients:
            print(f"Client {client.client_id} Local training")
            # Train (Anchoring happens inside)
            client.train(epochs=config.EPOCHS_PER_ROUND, server_prototypes=global_protos, round_num=round_num)
            
            # Compute new local prototypes
            c_protos = client.compute_prototypes()
            client_protos_list.append(c_protos)
            
        print(f"Server: Collecting client prototypes")
        
        # 4. Server Aggregation
        server.aggregate_prototypes(client_protos_list)
        
        # 5. Server Refinement
        server.refine_prototypes()
        
        round_end_time = time.time()
        logger.log_training_time(round_num, round_end_time - round_start_time)
        
        logger.log_round(round_num)
        
        # Evaluation
        if round_num % config.EVAL_FREQ == 0 or round_num == config.NUM_ROUNDS:
            print(f"\nEvaluation Round {round_num}")
            
            # Use local eval first (FedTGP style)
            local_eval = evaluate_all_clients_local([c.model for c in clients], client_test_loaders, device, global_protos, num_classes, client_info['classes'])
            print(f"Local Test: Weighted={local_eval['weighted_accuracy']:.4f}, Mean={local_eval['mean_accuracy']:.4f}")
            
            clients_metrics = []
            for i, client in enumerate(clients):
                 # FedPAGR Main has flexible logic: "if global_protos and len == num_classes"
                 if global_protos and len(global_protos) == num_classes:
                     metrics = evaluate_model_with_prototypes(client.model, test_loader, device, global_protos, num_classes)
                 else:
                     metrics = evaluate_model(client.model, test_loader, device, num_classes)
                 clients_metrics.append(metrics)

            if global_protos and len(global_protos) == num_classes:
                ensemble_metrics = evaluate_ensemble_with_prototypes([c.model for c in clients], test_loader, device, global_protos, num_classes)
            else:
                ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)

            print(f"Global Test Ensemble: Acc={ensemble_metrics['accuracy']:.4f}")
            
            logger.log_evaluation(round_num, clients_metrics, ensemble_metrics, local_metrics=local_eval)
            
            # Track best accuracy (using Weighted Local as per older logs)
            current_acc = local_eval['weighted_accuracy']
                
        if round_num % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("Metrics")
    print("="*80)
    
    global_protos = server.global_prototypes

    for i, (client, model_name) in enumerate(zip(clients, config.MODEL_TYPES)): 
        # config.MODEL_TYPES might be list of strings, but clients count might be larger.
        # Use simple indexing
        m_name = model_names[i % len(model_names)]
        
        print(f"\nClient {i} ({m_name}):")
        if global_protos and len(global_protos) == num_classes:
             metrics = evaluate_model_with_prototypes(client.model, test_loader, device, global_protos, num_classes)
             print(f"Prototype Inference Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        else:
             metrics = evaluate_model(client.model, test_loader, device, num_classes)
             print(f"Model Classifier Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"F1 (Macro):   {metrics['f1_macro']:.4f}")
        print(f"F1 (Weighted):{metrics['f1_weighted']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        
        local_classes = set(client_info['classes'][i])
        per_class = metrics['per_class_accuracy']
        print(f"\nPer-Class Accuracy:")
        print(f"  {'Class':<8} {'Acc':<8} {'Type':<10}")
        print(f"  {'-'*26}")
        for c in range(num_classes):
            class_type = "LOCAL" if c in local_classes else "non-local"
            print(f"  {c:<8} {per_class[c]:.4f}   {class_type}")
            
        local_accs = [per_class[c] for c in range(num_classes) if c in local_classes]
        nonlocal_accs = [per_class[c] for c in range(num_classes) if c not in local_classes]
        print(f"\n  Local avg:     {np.mean(local_accs):.4f}")
        print(f"  Non-local avg: {np.mean(nonlocal_accs):.4f}")

    print(f"\nEnsemble:")
    if global_protos and len(global_protos) == num_classes:
        ensemble_metrics = evaluate_ensemble_with_prototypes([c.model for c in clients], test_loader, device, global_protos, num_classes)
        print(f"Prototype Ensemble Accuracy:  {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    else:
        ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
        print(f"Model Ensemble Accuracy:      {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"F1 (Macro):   {ensemble_metrics['f1_macro']:.4f}")
    
    
    print("\n" + "="*80)
    print("Communication Costs")
    print("="*80)
    
    # Accurate Cost Calculation
    upload_bytes = num_classes * config.FEATURE_DIM * 4
    download_bytes = num_classes * config.FEATURE_DIM * 4
    bytes_per_round_per_client = upload_bytes + download_bytes
    total_bytes = bytes_per_round_per_client * config.NUM_CLIENTS * config.NUM_ROUNDS
    total_mb = total_bytes / (1024 * 1024)
    comm_efficiency = 0 / total_mb if total_mb > 0 else 0
    
    print(f"Bytes per round per client: {bytes_per_round_per_client:,.0f} bytes ({bytes_per_round_per_client/1024:.2f} KB)")
    print(f"Total communication:        {total_bytes:,.0f} bytes ({total_mb:.2f} MB)")
    print(f"Communication efficiency:   {comm_efficiency:.4f} (Accuracy/MB)")
    
    logger.log_communication(bytes_per_round_per_client, total_bytes, total_mb, comm_efficiency)
    logger.save_results()
    logger.plot_metrics()
    print("Completed")

if __name__ == "__main__":
    run()
