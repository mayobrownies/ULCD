import torch
import numpy as np
import random
import os
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import cv2
from PIL import Image
from . import config


class CachedTransformDataset(Dataset):

    def __init__(self, cached_tensors, labels):

        self.cached_tensors = cached_tensors
        self.labels = labels

    def __len__(self):
        return len(self.cached_tensors)

    def __getitem__(self, idx):
        return self.cached_tensors[idx], self.labels[idx]


class TransformDataset(Dataset):

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class GrayscaleTransform:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_pil = transforms.ToPILImage()(img)
            gray = img_pil.convert('L').convert('RGB')
            return transforms.ToTensor()(gray)
        return img


class EdgeDetectionTransform:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            return torch.from_numpy(edges_rgb.transpose(2, 0, 1)).float() / 255.0
        return img


class BlurTransform:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_pil = transforms.ToPILImage()(img)
            blurred = img_pil.filter(Image.FILTER.GaussianBlur(radius=2))
            return transforms.ToTensor()(blurred)
        return img


class NoiseTransform:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * self.std
            return torch.clamp(img + noise, 0, 1)
        return img


def get_transform_by_name(transform_name):
    base_transform = transforms.Compose([transforms.ToTensor()])

    transform_map = {
        'rgb': None,
        'grayscale': GrayscaleTransform(),
        'edge': EdgeDetectionTransform(),
        'blur': BlurTransform(),
        'color_jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        'noise': NoiseTransform(std=0.1)
    }

    if transform_name not in transform_map:
        raise ValueError(f"Unknown transform: {transform_name}. Available: {list(transform_map.keys())}")

    return transform_map[transform_name]


def get_transform_cache_path(transform_name, dataset_hash):
    cache_dir = config.TRANSFORM_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"transform_{transform_name}_{dataset_hash}.pt")


def compute_dataset_hash(dataset):
    if isinstance(dataset, Subset):
        indices_hash = hash(tuple(sorted(dataset.indices)))
    else:
        indices_hash = hash(len(dataset))
    return str(abs(indices_hash))[:16]


def cache_transformed_dataset(base_dataset, transform, transform_name):
    dataset_hash = compute_dataset_hash(base_dataset)
    cache_path = get_transform_cache_path(transform_name, dataset_hash)

    if os.path.exists(cache_path):
        print(f"Loading cached transform: {transform_name} from {cache_path}")
        cached_data = torch.load(cache_path)
        return CachedTransformDataset(cached_data['tensors'], cached_data['labels'])

    print(f"Pre-computing transform: {transform_name}")
    transformed_tensors = []
    labels = []

    temp_loader = DataLoader(base_dataset, batch_size=100, shuffle=False, num_workers=4)

    for imgs, lbls in temp_loader:
        if transform is not None:
            batch_transformed = []
            for img in imgs:
                transformed_img = transform(img)
                batch_transformed.append(transformed_img)
            imgs = torch.stack(batch_transformed)

        transformed_tensors.append(imgs)
        labels.append(lbls)

    all_tensors = torch.cat(transformed_tensors, dim=0)
    all_labels = torch.cat(labels, dim=0)

    torch.save({'tensors': all_tensors, 'labels': all_labels}, cache_path)
    print(f"Cached to: {cache_path}")

    return CachedTransformDataset(all_tensors, all_labels)


def get_label_heterogeneity_assignment(num_clients, num_classes=10):
    if config.LABEL_HETEROGENEITY == "none":
        return [list(range(num_classes)) for _ in range(num_clients)]

    elif config.LABEL_HETEROGENEITY == "partial_overlap":
        client_classes = []
        stride = config.LABEL_CLASSES_PER_CLIENT - config.LABEL_OVERLAP_SIZE

        for i in range(num_clients):
            start_idx = (i * stride) % num_classes
            classes = []
            for j in range(config.LABEL_CLASSES_PER_CLIENT):
                classes.append((start_idx + j) % num_classes)
            client_classes.append(sorted(classes))

        return client_classes

    elif config.LABEL_HETEROGENEITY == "non_overlapping":
        classes_per_client = num_classes // num_clients
        if classes_per_client == 0:
            raise ValueError(f"Cannot split {num_classes} classes across {num_clients} clients")

        client_classes = []
        class_list = list(range(num_classes))

        for i in range(num_clients):
            start = i * classes_per_client
            end = start + classes_per_client if i < num_clients - 1 else num_classes
            client_classes.append(class_list[start:end])

        return client_classes

    else:
        raise ValueError(f"Unknown LABEL_HETEROGENEITY: {config.LABEL_HETEROGENEITY}")


def get_data_transform_assignment(num_clients):
    if config.DATA_HETEROGENEITY == "none":
        return ['rgb'] * num_clients

    elif config.DATA_HETEROGENEITY == "transform":
        transforms = config.DATA_TRANSFORM_TYPES
        return [transforms[i % len(transforms)] for i in range(num_clients)]

    elif config.DATA_HETEROGENEITY == "mixed":
        if config.DATA_TRANSFORM_ASSIGNMENT and len(config.DATA_TRANSFORM_ASSIGNMENT) == num_clients:
            return config.DATA_TRANSFORM_ASSIGNMENT
        else:
            assignment = []
            half = num_clients // 2
            assignment.extend(['rgb'] * half)
            transforms = config.DATA_TRANSFORM_TYPES[1:]
            for i in range(num_clients - half):
                assignment.append(transforms[i % len(transforms)])
            return assignment

    else:
        raise ValueError(f"Unknown DATA_HETEROGENEITY: {config.DATA_HETEROGENEITY}")


def filter_dataset_by_classes(dataset, allowed_classes):
    allowed_classes_set = set(allowed_classes)
    filtered_indices = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label in allowed_classes_set:
            filtered_indices.append(idx)

    return filtered_indices


def create_client_datasets(base_dataset, num_clients, public_size, num_classes=10):
    client_label_classes = get_label_heterogeneity_assignment(num_clients, num_classes)
    client_transforms = get_data_transform_assignment(num_clients)

    print(f"\nHeterogeneity Configuration:")
    print(f"Label Heterogeneity: {config.LABEL_HETEROGENEITY}")
    print(f"Data Heterogeneity: {config.DATA_HETEROGENEITY}")
    print(f"\nClient Class Assignments:")
    for i, classes in enumerate(client_label_classes):
        print(f"Client {i}: Classes {classes} | Transform: {client_transforms[i]}")

    all_indices_by_class = {i: [] for i in range(num_classes)}
    for idx in range(len(base_dataset)):
        _, label = base_dataset[idx]
        all_indices_by_class[label].append(idx)

    for class_indices in all_indices_by_class.values():
        random.shuffle(class_indices)

    samples_per_class = public_size // num_classes
    public_indices = []
    for class_id in range(num_classes):
        public_indices.extend(all_indices_by_class[class_id][:samples_per_class])
    random.shuffle(public_indices)
    public_dataset = Subset(base_dataset, public_indices)

    print(f"\nPublic Dataset: {len(public_indices)} samples ({samples_per_class} per class, stratified)")

    public_set = set(public_indices)
    indices_by_class = {i: [] for i in range(num_classes)}
    for class_id in range(num_classes):
        for idx in all_indices_by_class[class_id]:
            if idx not in public_set:
                indices_by_class[class_id].append(idx)

    client_datasets = []

    for client_id in range(num_clients):
        allowed_classes = client_label_classes[client_id]
        client_indices = []

        for class_label in allowed_classes:
            class_samples = indices_by_class[class_label]

            if config.LABEL_HETEROGENEITY == "none":
                samples_per_client = len(class_samples) // num_clients
                start = client_id * samples_per_client
                end = start + samples_per_client if client_id < num_clients - 1 else len(class_samples)
                client_indices.extend(class_samples[start:end])
            else:
                if hasattr(config, 'SHARDS_PER_CLIENT') and config.SHARDS_PER_CLIENT > 0:
                    shard_size = len(class_samples) // config.SHARDS_PER_CLIENT
                    for shard_id in range(min(config.SHARDS_PER_CLIENT, 2)):
                        start = shard_id * shard_size
                        end = start + shard_size
                        client_indices.extend(class_samples[start:end])
                else:
                    client_indices.extend(class_samples)

        client_subset = Subset(base_dataset, client_indices)

        transform_name = client_transforms[client_id]
        transform = get_transform_by_name(transform_name)

        if transform is not None:
            if (config.CACHE_TRANSFORMS and
                transform_name in ['edge', 'blur', 'grayscale', 'noise']):
                client_dataset = cache_transformed_dataset(client_subset, transform, transform_name)
            else:
                client_dataset = TransformDataset(client_subset, transform)
        else:
            client_dataset = client_subset

        client_datasets.append(client_dataset)

    return client_datasets, public_dataset, client_label_classes, client_transforms


def get_public_dataset_tensors(public_loader):
    X_public_list = []
    y_public_list = []
    for x, y in public_loader:
        X_public_list.append(x)
        y_public_list.append(y)
    X_public = torch.cat(X_public_list, dim=0)
    y_public = torch.cat(y_public_list, dim=0)
    return (X_public, y_public)


def get_heterogeneous_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    cifar_train = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    cifar_test = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    client_datasets, public_dataset, client_classes, client_transforms = create_client_datasets(
        cifar_train,
        config.NUM_CLIENTS,
        config.PUBLIC_SIZE,
        num_classes=10
    )

    dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'num_workers': config.DATALOADER_NUM_WORKERS,
        'pin_memory': config.DATALOADER_PIN_MEMORY,
    }

    if (config.DATALOADER_NUM_WORKERS > 0 and
        config.DATALOADER_PERSISTENT_WORKERS and
        not (config.USE_MULTI_GPU and len(config.GPU_IDS) > 1)):
        dataloader_kwargs['persistent_workers'] = True
    elif config.USE_MULTI_GPU and len(config.GPU_IDS) > 1:
        print("Note: persistent_workers disabled")

    client_loaders = [
        DataLoader(dataset, shuffle=True, **dataloader_kwargs)
        for dataset in client_datasets
    ]

    public_loaders = [
        DataLoader(public_dataset, shuffle=False, **dataloader_kwargs)
        for _ in range(config.NUM_CLIENTS)
    ]

    test_loader = DataLoader(
        cifar_test,
        shuffle=False,
        **dataloader_kwargs
    )

    client_info = {
        'classes': client_classes,
        'transforms': client_transforms
    }

    return client_loaders, public_loaders, test_loader, client_info
