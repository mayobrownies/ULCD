import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .models import CNNModel, MLPModel, ResNetModel
from .fedproto_client import FedProtoClient
from .fedproto_server import FedProtoServer
from .logger import FLLogger
from .data_utils import get_heterogeneous_dataloaders
from . import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
            log_probs, _ = model(x, return_protos=True, use_consensus=False)
            preds = log_probs.argmax(dim=1).cpu().numpy()
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
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
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
                log_probs, _ = model(x, return_protos=True, use_consensus=False)
                probs = torch.exp(log_probs)
                ensemble_probs.append(probs)

            avg_probs = torch.stack(ensemble_probs).mean(dim=0)
            preds = avg_probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro')
    }


def evaluate_model_with_prototypes(model, test_loader, device, global_prototypes, local_classes=None, num_classes=10):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, features = model(x, return_protos=True, use_consensus=False)
            all_features.append(features)
            all_labels.extend(y.numpy())

    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)

    local_mask = np.isin(all_labels, list(local_classes))
    local_features = all_features[torch.tensor(local_mask)]
    local_labels = all_labels[local_mask]

    if len(local_labels) > 0:
        distances = torch.full((len(local_labels), num_classes), 1e6, device=device)
        for c in local_classes:
            if c in global_prototypes:
                proto = global_prototypes[c].to(device)
                distances[:, c] = ((local_features - proto.unsqueeze(0)) ** 2).mean(dim=1)

        preds = distances.argmin(dim=1).cpu().numpy()

        per_class_acc = {}
        for c in local_classes:
            mask = local_labels == c
            if mask.sum() > 0:
                per_class_acc[c] = (preds[mask] == c).mean()
            else:
                per_class_acc[c] = 0.0

        return {
            'accuracy': accuracy_score(local_labels, preds),
            'f1_macro': f1_score(local_labels, preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(local_labels, preds, average='weighted', zero_division=0),
            'precision': precision_score(local_labels, preds, average='macro', zero_division=0),
            'recall': recall_score(local_labels, preds, average='macro', zero_division=0),
            'per_class_accuracy': per_class_acc
        }
    else:
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'per_class_accuracy': {}
        }


def evaluate_ensemble_with_prototypes(models, test_loader, device, global_prototypes, num_classes=10):
    for model in models:
        model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            batch_size = x.shape[0]

            ensemble_features = None
            for model in models:
                _, features = model(x, return_protos=True, use_consensus=False)
                if ensemble_features is None:
                    ensemble_features = features
                else:
                    ensemble_features = ensemble_features + features
            ensemble_features = ensemble_features / len(models)

            distances = torch.full((batch_size, num_classes), 1e6, device=device)
            for c in range(num_classes):
                if c in global_prototypes:
                    proto = global_prototypes[c].to(device)
                    distances[:, c] = ((ensemble_features - proto.unsqueeze(0)) ** 2).mean(dim=1)

            preds = distances.argmin(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }


def save_checkpoint(clients, server, round_num, best_acc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'round': round_num,
        'best_acc': best_acc,
        'prototypes': server.global_prototypes,
        'models': {i: c.model.state_dict() for i, c in enumerate(clients)}
    }
    torch.save(checkpoint, f"{output_dir}/checkpoint_r{round_num}.pt")
    print(f"Saved at round {round_num}")


def run():
    set_seed(config.SEED)
    if config.USE_CUDA and torch.cuda.is_available():
        torch.cuda.set_device(config.GPU_IDS[0])
    device = torch.device(f"cuda:{config.GPU_IDS[0]}" if (torch.cuda.is_available() and config.USE_CUDA) else "cpu")

    logger = FLLogger(output_dir=config.OUTPUT_DIR, experiment_name="FedProto")

    print(f"Device: {device}")
    print(f"Models: {', '.join(config.MODEL_TYPES)}")
    print(f"Feature Dimension: {config.FEATURE_DIM}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    lr_decay_status = "Disabled" if config.LR_DECAY_GAMMA == 1.0 else f"Step={config.LR_DECAY_STEP}, Gamma={config.LR_DECAY_GAMMA}"
    print(f"LR Decay: {lr_decay_status}")
    print(f"Proto Weight: {config.PROTOTYPE_WEIGHT}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Epochs per Round: {config.EPOCHS_PER_ROUND}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print(f"Label Heterogeneity: {config.LABEL_HETEROGENEITY}")
    print(f"Eval Frequency: Every {config.EVAL_FREQ} rounds")
    print(f"Checkpoint Frequency: Every {config.CHECKPOINT_FREQ} rounds")
    print("="*80)

    client_loaders, _, test_loader, client_info = get_heterogeneous_dataloaders()

    logger.set_client_classes(client_info['classes'])

    models = [
        CNNModel(feature_dim=config.FEATURE_DIM),
        MLPModel(feature_dim=config.FEATURE_DIM),
        ResNetModel(feature_dim=config.FEATURE_DIM)
    ]
    clients = [FedProtoClient(model.to(device), cl, i) for i, (model, cl) in enumerate(zip(models, client_loaders))]
    server = FedProtoServer()

    best_acc = 0.0
    best_round = 0

    for round_num in range(1, config.NUM_ROUNDS + 1):
        print(f"\n{'='*80}")
        print(f"Round {round_num}/{config.NUM_ROUNDS}")
        print(f"{'='*80}")

        server.clear()

        prototypes = {}
        if server.global_prototypes:
            prototypes = server.broadcast()

        print(f"\nTraining")
        local_protos = {}
        for client in clients:
            client.train(epochs=config.EPOCHS_PER_ROUND, server_prototypes=prototypes, round_num=round_num)
            agg_protos = client.compute_prototypes()
            local_protos[client.client_id] = agg_protos

        print(f"\nAggregation")
        server.aggregate_prototypes(local_protos)
        print(f"Aggregated prototypes for {len(server.global_prototypes)} classes")

        logger.log_round(round_num)

        if round_num % config.EVAL_FREQ == 0 or round_num == config.NUM_ROUNDS:
            print(f"\nEvaluation Round {round_num}")
            clients_metrics = []
            for i, client in enumerate(clients):
                local_classes = set(client_info['classes'][i])
                metrics = evaluate_model_with_prototypes(client.model, test_loader, device, server.global_prototypes, local_classes=local_classes)
                clients_metrics.append(metrics)
                print(f"Client {i}: Acc={metrics['accuracy']:.4f} (local classes only)")

            ensemble_metrics = evaluate_ensemble_with_prototypes(
                [c.model for c in clients], test_loader, device, server.global_prototypes
            )
            print(f"Ensemble: Acc={ensemble_metrics['accuracy']:.4f}")

            logger.log_evaluation(round_num, clients_metrics, ensemble_metrics)
            logger.check_convergence(round_num)

            if ensemble_metrics['accuracy'] > best_acc:
                best_acc = ensemble_metrics['accuracy']
                best_round = round_num
                save_checkpoint(clients, server, round_num, best_acc, config.OUTPUT_DIR + "/best_fedproto")
                print(f"[NEW BEST] Accuracy: {best_acc:.4f}")

        if round_num % config.CHECKPOINT_FREQ == 0:
            save_checkpoint(clients, server, round_num, best_acc, config.OUTPUT_DIR + "/checkpoints_fedproto")

        if round_num % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("Final Metrics")
    print("="*80)

    for i, (client, model_name) in enumerate(zip(clients, config.MODEL_TYPES)):
        local_classes = set(client_info['classes'][i])
        metrics = evaluate_model_with_prototypes(client.model, test_loader, device, server.global_prototypes, local_classes=local_classes)

        print(f"\nClient {i} ({model_name}):")
        print(f"Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) (local classes only)")
        print(f"F1 (Macro): {metrics['f1_macro']:.4f}")

        # Per-class accuracy breakdown
        per_class = metrics['per_class_accuracy']
        print(f"\nPer-Class Accuracy (local classes: {sorted(local_classes)}):")
        print(f"  {'Class':<8} {'Acc':<8}")
        print(f"  {'-'*16}")
        for c in sorted(local_classes):
            if c in per_class:
                print(f"  {c:<8} {per_class[c]:.4f}")

    print(f"\n{'='*40}")
    print(f"Ensemble Metrics")
    print(f"{'='*40}")

    ensemble_metrics = evaluate_ensemble_with_prototypes(
        [c.model for c in clients], test_loader, device, server.global_prototypes
    )

    print(f"Accuracy:     {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"F1 (Macro):   {ensemble_metrics['f1_macro']:.4f}")
    print(f"F1 (Weighted):{ensemble_metrics['f1_weighted']:.4f}")
    print(f"Precision:    {ensemble_metrics['precision']:.4f}")
    print(f"Recall:       {ensemble_metrics['recall']:.4f}")

    print(f"\nBest Ensemble Accuracy: {best_acc:.4f} at Round {best_round}")

    print("\n" + "="*80)
    print("Communication Costs")
    print("="*80)
    upload_bytes = 10 * config.FEATURE_DIM * 4
    download_bytes = 10 * config.FEATURE_DIM * 4
    bytes_per_round_per_client = upload_bytes + download_bytes
    total_bytes = bytes_per_round_per_client * config.NUM_CLIENTS * config.NUM_ROUNDS
    total_mb = total_bytes / (1024 * 1024)
    comm_efficiency = ensemble_metrics['accuracy'] / total_mb

    print(f"Bytes per round per client: {bytes_per_round_per_client:,} bytes ({bytes_per_round_per_client/1024:.2f} KB)")
    print(f"Total communication:        {total_bytes:,} bytes ({total_mb:.2f} MB)")
    print(f"Communication efficiency:   {comm_efficiency:.4f} (Accuracy/MB)")

    logger.log_communication(bytes_per_round_per_client, total_bytes, total_mb, comm_efficiency)
    logger.log_best(best_acc, best_round)

    logger.save_results()
    logger.plot_metrics()

    print("Completed")


if __name__ == "__main__":
    run()
