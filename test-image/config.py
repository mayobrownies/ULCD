# Data Configuration
DATA_DIR = "./data"
DATASET = "cifar10" # 'cifar10', 'cifar100', or 'tiny_imagenet'"
NUM_CLIENTS = 20
PUBLIC_SIZE = 0
SHARDS_PER_CLIENT = 2
BATCH_SIZE = 32

# Training Configuration
NUM_ROUNDS = 2
EPOCHS_PER_ROUND = 1
LEARNING_RATE = 0.01
SGD_MOMENTUM = 0.9
JOIN_RATIO = 1.0

# Learning Rate Decay
LR_DECAY_STEP = 999
LR_DECAY_GAMMA = 0.7

# Model Configuration
MODEL_TYPES = ["CNN", "MLP", "ResNet18", "GoogLeNet", "MobileNet"]
FEATURE_DIM = 512

# Prototype Alignment
PROTOTYPE_WEIGHT = 1.0
PROTOTYPE_WARMUP_RATE = 0.01

# FedMD-specific Configuration
N_ALIGNMENT = 1000
N_LOGITS_MATCHING_ROUND = 1
N_PRIVATE_TRAINING_ROUND = 2
PRIVATE_TRAINING_BATCHSIZE = 32

# FedTGP-specific Configuration
FEDTGP_SERVER_EPOCHS = 1000
FEDTGP_SERVER_BATCH_SIZE = 16
FEDTGP_MARGIN_THRESHOLD = 100.0


# FedPAGR-specific Configuration
FEDPAGR_BETA = 0.1 # Client Global Regularization
FEDPAGR_LAMBDA_S = 0.5 # Server Separation Weight (was 1.0, ablation lsep_0.5 best)
FEDPAGR_LAMBDA_T = 0.0 # Server Temporal Weight (was 0.5, ablation ltemp_0.0 best)
FEDPAGR_MARGIN = 0.3 # Server Separation Margin
FEDPAGR_REFINE_LR = 0.01 # Server Refinement Learning Rate
FEDPAGR_REFINE_STEPS = 5 # Server Refinement Steps
FEDPAGR_ENTROPY_WEIGHT = 0.1 # Client Uniform Entropy Weight (was 0.01, ablation entropy_0.1 best)

# Long Run Configuration
CHECKPOINT_FREQ = 20
EVAL_FREQ = 20
SEED = 42

# Performance Reporting
FINAL_AVERAGE_WINDOW = 5  # Average metrics over last N evaluations for steady-state performance

# Convergence Detection
CONVERGENCE_WINDOW = 10
CONVERGENCE_THRESHOLD = 0.01

# Logging Configuration
OUTPUT_DIR = "./fl_plots"
EXPERIMENT_NAME = "hetero_fl"

# Device Configuration
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
GPU_IDS = [0]

# Transform Caching
CACHE_TRANSFORMS = True
TRANSFORM_CACHE_DIR = "./data/transform_cache"

# Controls how classes are distributed across clients
LABEL_HETEROGENEITY = "dirichlet"  # "none", "partial_overlap", "non_overlapping", "dirichlet"

# For "partial_overlap": Each client gets LABEL_CLASSES_PER_CLIENT classes,
# with consecutive clients having LABEL_OVERLAP_SIZE classes in common
LABEL_CLASSES_PER_CLIENT = 5  # How many classes each client has
LABEL_OVERLAP_SIZE = 2  # How many classes overlap between consecutive clients

# For "non_overlapping": Classes are split evenly across clients with no overlap
# Example with 10 classes, 5 clients: each gets 2 unique classes

# For "dirichlet": Use Dirichlet distribution for realistic non-IID data
# alpha controls heterogeneity: 0.1 = very non-IID, 1.0 = moderate, 100 = nearly IID
DIRICHLET_ALPHA = 0.1

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
