import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .models import CNNModel, MLPModel, ResNetModel
from .ulcd_client import ULCDClient
from .ulcd_server import ULCDServer
from .logger import FLLogger
from .data_utils import get_heterogeneous_dataloaders
from .multi_gpu_utils import parallel_train_clients
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
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()
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


def evaluate_model_with_prototypes(model, test_loader, device, global_prototypes, num_classes=10):
    model.eval()
    all_preds = []
    all_labels = []

    proto_tensor = torch.stack([global_prototypes[c].to(device) for c in range(num_classes)])

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
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

    logger = FLLogger(output_dir=config.OUTPUT_DIR, experiment_name="ULCD (Contrastive Consensus)")

    print(f"Device: {device}")
    print(f"Models: {', '.join(config.MODEL_TYPES)}")
    print(f"Feature Dimension: {config.FEATURE_DIM}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    lr_decay_status = "Disabled" if config.LR_DECAY_GAMMA == 1.0 else f"Step={config.LR_DECAY_STEP}, Gamma={config.LR_DECAY_GAMMA}"
    print(f"LR Decay: {lr_decay_status}")
    print(f"Proto Weight: {config.PROTOTYPE_WEIGHT}")
    print(f"Contrastive Weight: {config.CONTRASTIVE_WEIGHT}")
    print(f"Contrastive Temp: {config.CONTRASTIVE_TEMP}")
    print(f"EMA Momentum: {config.ULCD_EMA_MOMENTUM}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Epochs per Round: {config.EPOCHS_PER_ROUND}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print(f"Label Heterogeneity: {config.LABEL_HETEROGENEITY}")
    print(f"Public Alignment: {config.ULCD_USE_PUBLIC_ALIGNMENT}")
    if config.ULCD_USE_PUBLIC_ALIGNMENT:
        print(f"  Public Align Epochs: {config.ULCD_PUBLIC_ALIGNMENT_EPOCHS}")
        print(f"  Public Align Batch Size: {config.ULCD_PUBLIC_ALIGNMENT_BATCH_SIZE}")
    print(f"EMA Smoothing: {config.ULCD_USE_EMA}")
    print(f"Class Weighting: {config.ULCD_USE_CLASS_WEIGHTING}")
    print(f"Contrastive Loss: {config.ULCD_USE_CONTRASTIVE}")
    print(f"Eval Frequency: Every {config.EVAL_FREQ} rounds")
    print(f"Checkpoint Frequency: Every {config.CHECKPOINT_FREQ} rounds")
    print("="*80)

    client_loaders, public_loaders, test_loader, client_info = get_heterogeneous_dataloaders()

    logger.set_client_classes(client_info['classes'])

    models = [
        CNNModel(num_classes=10, feature_dim=config.FEATURE_DIM),
        MLPModel(num_classes=10, feature_dim=config.FEATURE_DIM),
        ResNetModel(num_classes=10, feature_dim=config.FEATURE_DIM)
    ]

    clients = [
        ULCDClient(model.to(device), cl, pl, i)
        for i, (model, cl, pl) in enumerate(zip(models, client_loaders, public_loaders))
    ]

    server = ULCDServer(latent_dim=config.FEATURE_DIM, num_classes=10,
                       ema_momentum=config.ULCD_EMA_MOMENTUM)

    best_acc = 0.0
    best_round = 0

    for round_num in range(1, config.NUM_ROUNDS + 1):
        print(f"\n{'='*80}")
        print(f"Round {round_num}/{config.NUM_ROUNDS}")
        print(f"{'='*80}")

        server.clear()

        prototypes = None
        if server.global_prototypes:
            prototypes, _ = server.broadcast()

        if config.USE_MULTI_GPU and len(config.GPU_IDS) > 1:
            print(f"\nParallel client training on {len(config.GPU_IDS)} GPUs")
            clients = parallel_train_clients(
                clients,
                epochs=config.EPOCHS_PER_ROUND,
                server_prototypes=prototypes,
                round_num=round_num
            )
        else:
            for client in clients:
                print(f"\nClient {client.client_id} Local training")
                client.train(epochs=config.EPOCHS_PER_ROUND,
                            server_prototypes=prototypes,
                            round_num=round_num)

        print(f"\nServer: Collecting client prototypes")
        client_summaries = [client.compute_prototypes() for client in clients]
        server.aggregate_prototypes(client_summaries, round_num=round_num)

        logger.log_round(round_num)

        if round_num % config.EVAL_FREQ == 0 or round_num == config.NUM_ROUNDS:
            print(f"\nEvaluation Round {round_num}")
            clients_metrics = []
            global_protos = server.global_prototypes

            for i, client in enumerate(clients):
                local_classes = set(client_info['classes'][i])

                if global_protos and len(global_protos) == 10:
                    proto_metrics = evaluate_model_with_prototypes(client.model, test_loader, device, global_protos)
                    clients_metrics.append(proto_metrics)
                    proto_per_class = proto_metrics['per_class_accuracy']
                    proto_local = np.mean([proto_per_class[c] for c in local_classes])
                    proto_nonlocal = np.mean([proto_per_class[c] for c in range(10) if c not in local_classes])
                    print(f"{config.MODEL_TYPES[i]}: Acc={proto_metrics['accuracy']:.4f}, F1={proto_metrics['f1_macro']:.4f}")
                    print(f"Local classes {sorted(local_classes)}: {proto_local:.4f} avg")
                    print(f"Non-local classes {sorted(set(range(10)) - local_classes)}: {proto_nonlocal:.4f} avg")
                else:
                    # Fallback to classifier-based if no prototypes
                    metrics = evaluate_model(client.model, test_loader, device)
                    clients_metrics.append(metrics)
                    per_class = metrics['per_class_accuracy']
                    local_accs = [per_class[c] for c in range(10) if c in local_classes]
                    nonlocal_accs = [per_class[c] for c in range(10) if c not in local_classes]
                    local_avg = np.mean(local_accs) if local_accs else 0
                    nonlocal_avg = np.mean(nonlocal_accs) if nonlocal_accs else 0
                    print(f"{config.MODEL_TYPES[i]}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
                    print(f"Local classes {sorted(local_classes)}: {local_avg:.4f} avg")
                    print(f"Non-local classes {sorted(set(range(10)) - local_classes)}: {nonlocal_avg:.4f} avg")

            if global_protos and len(global_protos) == 10:
                ensemble_metrics = evaluate_ensemble_with_prototypes([c.model for c in clients], test_loader, device, global_protos)
            else:
                ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
            print(f"Ensemble: Acc={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1_macro']:.4f}")

            logger.log_evaluation(round_num, clients_metrics, ensemble_metrics)
            logger.check_convergence(round_num)

            if ensemble_metrics['accuracy'] > best_acc:
                best_acc = ensemble_metrics['accuracy']
                best_round = round_num
                save_checkpoint(clients, server, round_num, best_acc, config.OUTPUT_DIR + "/best_ulcd")
                print(f"Accuracy: {best_acc:.4f}")

        if round_num % config.CHECKPOINT_FREQ == 0:
            save_checkpoint(clients, server, round_num, best_acc, config.OUTPUT_DIR + "/checkpoints_ulcd")

        if round_num % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("Metrics")
    print("="*80)

    global_protos = server.global_prototypes

    for i, (client, model_name) in enumerate(zip(clients, config.MODEL_TYPES)):
        print(f"\nClient {i} ({model_name}):")

        if global_protos and len(global_protos) == 10:
            metrics = evaluate_model_with_prototypes(client.model, test_loader, device, global_protos)
        else:
            metrics = evaluate_model(client.model, test_loader, device)

        print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"F1 (Macro):   {metrics['f1_macro']:.4f}")
        print(f"F1 (Weighted):{metrics['f1_weighted']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")

        # Per-class accuracy breakdown
        local_classes = set(client_info['classes'][i])
        per_class = metrics['per_class_accuracy']
        print(f"\nPer-Class Accuracy:")
        print(f"  {'Class':<8} {'Acc':<8} {'Type':<10}")
        print(f"  {'-'*26}")
        for c in range(10):
            class_type = "LOCAL" if c in local_classes else "non-local"
            print(f"  {c:<8} {per_class[c]:.4f}   {class_type}")

        local_accs = [per_class[c] for c in range(10) if c in local_classes]
        nonlocal_accs = [per_class[c] for c in range(10) if c not in local_classes]
        print(f"\n  Local avg:     {np.mean(local_accs):.4f}")
        print(f"  Non-local avg: {np.mean(nonlocal_accs):.4f}")

    print(f"\nEnsemble:")
    if global_protos and len(global_protos) == 10:
        ensemble_metrics = evaluate_ensemble_with_prototypes([c.model for c in clients], test_loader, device, global_protos)
    else:
        ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
    print(f"Accuracy:     {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"F1 (Macro):   {ensemble_metrics['f1_macro']:.4f}")
    print(f"F1 (Weighted):{ensemble_metrics['f1_weighted']:.4f}")
    print(f"Precision:    {ensemble_metrics['precision']:.4f}")
    print(f"Recall:       {ensemble_metrics['recall']:.4f}")

    print(f"\nBest Ensemble Accuracy: {best_acc:.4f} at Round {best_round}")

    print("\n" + "="*80)
    print("Communication Costs")
    print("="*80)
    upload_bytes = 10 * config.FEATURE_DIM * 4 + 10 * 4
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
