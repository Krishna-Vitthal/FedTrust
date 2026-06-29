import random
import numpy as np
import torch

# =====================================================
# Federated Learning Configuration
# =====================================================

NUM_CLIENTS = 10

ROUNDS = 20

LOCAL_EPOCHS = 1

BATCH_SIZE = 32

LEARNING_RATE = 0.001

# =====================================================
# Dataset Split
# =====================================================

# "iid"
# "noniid"

DATA_SPLIT = "iid"

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
# Trust Configuration (FedTrust)
# =====================================================

USE_TRUST = True

# -----------------------------
# Initial Trust
# -----------------------------

INITIAL_TRUST = 0.5

MIN_TRUST = 0.05

MAX_TRUST = 1.0

# -----------------------------
# EMA
# -----------------------------

ALPHA = 0.7

# -----------------------------
# Behavior Score Weights
# -----------------------------

SIMILARITY_WEIGHT = 0.35

CONSISTENCY_WEIGHT = 0.25

ALIGNMENT_WEIGHT = 0.20

NORM_WEIGHT = 0.20

# -----------------------------
# Peer Grouping
# -----------------------------

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