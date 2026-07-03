import random
import numpy as np
import torch

# =====================================================
# Federated Learning Configuration
# =====================================================

NUM_CLIENTS = 10
RECOVERY_THRESHOLD = 0.80
RECOVERY_ROUNDS = 5
RECOVERY_STEP = 0.02

RUN_ALL_ATTACKS = True

BENCHMARK_MODE = "comparison"

# Options:
# "comparison"
# "ablation"
# "sensitivity"
# "stress"
# "runtime"
# "diagnostics"
# "full"

RESULTS_DIR = "results"

EXPERIMENT_TAG = "default"

SAVE_DIAGNOSTICS = False

USE_TRIMMED_MEAN = True

TRIM_RATIO = 0.20

ROUNDS = 20

LOCAL_EPOCHS = 1

BATCH_SIZE = 32

LEARNING_RATE = 0.001

# =====================================================
# Dataset Split
# =====================================================

# Options:
# "iid"
# "noniid"
# "dirichlet"

DATA_SPLIT = "dirichlet"

# Dirichlet concentration parameter
# Smaller = more heterogeneous
# Larger = closer to IID

DIRICHLET_ALPHA = 0.5

# =====================================================
# Attack Configuration
# =====================================================

MALICIOUS_CLIENTS = 2

ATTACK_TYPE = "mixed"

ATTACK_SCALE = 8

GAUSSIAN_STD = 1.0

ATTACK_POOL = [
    "sign",
    "scaling",
    "gaussian",
    "random",
    "zero",
]

# =====================================================
# Trust Configuration
# =====================================================

USE_TRUST = True

# =====================================================
# Adaptive Trust Filtering
# =====================================================

USE_TRUST_FILTER = True

TRUST_FILTER_MODE = "adaptive"
# Options:
# "adaptive"
# "fixed"

FIXED_TRUST_THRESHOLD = 0.50

STD_FACTOR = 0.50

INITIAL_TRUST = 0.5

MIN_TRUST = 0.05

MAX_TRUST = 1.0

ALPHA = 0.7

RECOVERY_THRESHOLD = 0.6

RECOVERY_ROUNDS = 3

RECOVERY_STEP = 0.05

SIMILARITY_WEIGHT = 0.35

CONSISTENCY_WEIGHT = 0.25

ALIGNMENT_WEIGHT = 0.20

NORM_WEIGHT = 0.20

PEER_SIMILARITY_THRESHOLD = 0.70

# =====================================================
# Device
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# Random Seed
# =====================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False