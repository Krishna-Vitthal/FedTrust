import random

import numpy as np
import torch

# ----------------------------------
# Federated Learning Configuration
# ----------------------------------

NUM_CLIENTS = 10

ROUNDS = 20

LOCAL_EPOCHS = 1

BATCH_SIZE = 32

LEARNING_RATE = 0.001

# ----------------------------------
# Data Split Configuration
# ----------------------------------

DATA_SPLIT = "iid"

# ----------------------------------
# Attack Configuration
# ----------------------------------

MALICIOUS_CLIENTS = 2

# Options:
# "sign"
# "scaling"
# "gaussian"
# "random"
# "zero"
# "mixed"

ATTACK_TYPE = "mixed"

ATTACK_POOL = (
    "sign",
    "scaling",
    "gaussian",
    "random",
    "zero"
)

ATTACK_SCALE = 8

GAUSSIAN_STD = 1.0

# ----------------------------------
# Device
# ----------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ----------------------------------
# Random Seed
# ----------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False