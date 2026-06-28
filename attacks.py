import copy
import random

import torch

from config import ATTACK_SCALE, ATTACK_POOL, GAUSSIAN_STD


# -------------------------------------------------------
# Sign Flipping
# -------------------------------------------------------

def sign_flipping(client_update):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        malicious_update["weights"][key] *= -1

    return malicious_update


# -------------------------------------------------------
# Model Scaling Attack (Strong)
# -------------------------------------------------------

def model_scaling(client_update):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        malicious_update["weights"][key] *= -ATTACK_SCALE

    return malicious_update


# -------------------------------------------------------
# Gaussian Noise Attack
# -------------------------------------------------------

def gaussian_noise(client_update):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        noise = (
            torch.randn_like(
                malicious_update["weights"][key]
            ) * GAUSSIAN_STD
        )

        malicious_update["weights"][key] += noise

    return malicious_update


# -------------------------------------------------------
# Random Model Attack
# -------------------------------------------------------

def random_weights(client_update):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        malicious_update["weights"][key] = (
            torch.randn_like(malicious_update["weights"][key]) * ATTACK_SCALE
        )

    return malicious_update


# -------------------------------------------------------
# Zero Model Attack
# -------------------------------------------------------

def zero_weights(client_update):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        malicious_update["weights"][key] = torch.zeros_like(
            malicious_update["weights"][key]
        )

    return malicious_update


# -------------------------------------------------------
# Dispatcher
# -------------------------------------------------------

def apply_attack(client_update, attack_type):

    if attack_type == "mixed":

        attack_type = random.choice(ATTACK_POOL)

    if attack_type == "sign":

        return sign_flipping(client_update)

    elif attack_type == "scaling":

        return model_scaling(client_update)

    elif attack_type == "gaussian":

        return gaussian_noise(client_update)

    elif attack_type == "random":

        return random_weights(client_update)

    elif attack_type == "zero":

        return zero_weights(client_update)

    else:

        return client_update