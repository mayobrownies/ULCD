# Data Configuration
DATA_DIR = "./data"
NUM_CLIENTS = 3
PUBLIC_SIZE = 1000
SHARDS_PER_CLIENT = 2
BATCH_SIZE = 256

# Training Configuration
NUM_ROUNDS = 100
EPOCHS_PER_ROUND = 3
LEARNING_RATE = 0.001

# Learning Rate Decay
LR_DECAY_STEP = 999 # disabled
LR_DECAY_GAMMA = 0.7

# Model Configuration
MODEL_TYPES = ["CNN", "MLP", "ResNet"]
FEATURE_DIM = 256

# Prototype Alignment
PROTOTYPE_WEIGHT = 1.0
PROTOTYPE_WARMUP_RATE = 0.01

# Contrastive Learning (ULCD only)
CONTRASTIVE_TEMP = 0.2
CONTRASTIVE_WEIGHT = 0.3

# FedMD-specific Configuration
N_ALIGNMENT = 1000
N_LOGITS_MATCHING_ROUND = 1
N_PRIVATE_TRAINING_ROUND = 2
PRIVATE_TRAINING_BATCHSIZE = 128

# ULCD-specific Configuration
ULCD_EMA_MOMENTUM = 0.9

ULCD_USE_PUBLIC_ALIGNMENT = True
ULCD_PUBLIC_ALIGNMENT_EPOCHS = 2
ULCD_PUBLIC_ALIGNMENT_BATCH_SIZE = 64
ULCD_PUBLIC_ALIGNMENT_LR = 0.001 
ULCD_USE_EMA = False
ULCD_USE_WARMUP = False
ULCD_USE_CLASS_WEIGHTING = False
ULCD_USE_CONTRASTIVE = True

# Mixed-specific Configuration
MIXED_DISTILL_WEIGHT = 0.1
MIXED_USE_LOGITS = True

# Long Run Configuration
CHECKPOINT_FREQ = 20
EVAL_FREQ = 5
SEED = 42

# Performance Reporting
FINAL_AVERAGE_WINDOW = 10  # Average metrics over last N evaluations for steady-state performance

# Convergence Detection
CONVERGENCE_WINDOW = 10
CONVERGENCE_THRESHOLD = 0.01

# Logging Configuration
OUTPUT_DIR = "./fl_plots"
EXPERIMENT_NAME = "hetero_fl"

# Device Configuration
USE_CUDA = True

# DataLoader Performance
DATALOADER_NUM_WORKERS = 8
DATALOADER_PIN_MEMORY = True
DATALOADER_PERSISTENT_WORKERS = True

# Mixed Precision Training
USE_AMP = False

# Multi-GPU Configuration
USE_MULTI_GPU = False
GPU_IDS = [1]

# Transform Caching
CACHE_TRANSFORMS = True
TRANSFORM_CACHE_DIR = "./data/transform_cache"

# Controls how classes are distributed across clients
LABEL_HETEROGENEITY = "partial_overlap"  # "none", "partial_overlap", "non_overlapping"

# For "partial_overlap": Each client gets LABEL_CLASSES_PER_CLIENT classes,
# with consecutive clients having LABEL_OVERLAP_SIZE classes in common
LABEL_CLASSES_PER_CLIENT = 5  # How many classes each client has
LABEL_OVERLAP_SIZE = 2  # How many classes overlap between consecutive clients

# For "non_overlapping": Classes are split evenly across clients with no overlap
# Example with 10 classes, 5 clients: each gets 2 unique classes

# Controls whether clients see different "types" of data (visual variations)
DATA_HETEROGENEITY = "none"  # "none", "transform", "mixed"

# For "transform": Each client gets a different visual transformation
# Available transforms: "rgb", "grayscale", "edge", "blur", "color_jitter", "noise"
# Clients will be assigned transforms in order from this list
DATA_TRANSFORM_TYPES = ["rgb", "grayscale", "edge", "blur", "color_jitter"]

# For "mixed": Some clients share transforms, some have unique
# Format: List of transform names, one per client (can repeat)
# Example: ["rgb", "rgb", "grayscale", "edge", "edge"] means clients 0,1 share RGB, 3,4 share edge
DATA_TRANSFORM_ASSIGNMENT = None  # If None, auto-assign from DATA_TRANSFORM_TYPES
