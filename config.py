import random
import numpy as np
import torch

# ----------------------------
# Federated Learning Parameters
# ----------------------------

NUM_CLIENTS = 10

ROUNDS = 20

LOCAL_EPOCHS = 1

BATCH_SIZE = 32

LEARNING_RATE = 0.001

# ----------------------------
# Attack Parameters
# ----------------------------

MALICIOUS_CLIENTS = 2

# ----------------------------
# Device
# ----------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ----------------------------
# Reproducibility
# ----------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False