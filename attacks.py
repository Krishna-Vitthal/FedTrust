import copy
import random
import torch


# -------------------------------------------------------
# Sign Flipping Attack
# -------------------------------------------------------

def sign_flipping(client_update):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        malicious_update["weights"][key] *= -1

    return malicious_update


# -------------------------------------------------------
# Gaussian Noise Attack
# -------------------------------------------------------

def gaussian_noise(client_update, std=0.5):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        noise = torch.randn_like(
            malicious_update["weights"][key]
        ) * std

        malicious_update["weights"][key] += noise

    return malicious_update


# -------------------------------------------------------
# Random Weight Attack
# -------------------------------------------------------

def random_weights(client_update):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        malicious_update["weights"][key] = torch.randn_like(
            malicious_update["weights"][key]
        )

    return malicious_update


# -------------------------------------------------------
# Zero Weight Attack
# -------------------------------------------------------

def zero_weights(client_update):

    malicious_update = copy.deepcopy(client_update)

    for key in malicious_update["weights"]:

        malicious_update["weights"][key] = torch.zeros_like(
            malicious_update["weights"][key]
        )

    return malicious_update


# -------------------------------------------------------
# Attack Dispatcher
# -------------------------------------------------------

def apply_attack(client_update, attack_type):

    if attack_type == "sign":

        return sign_flipping(client_update)

    elif attack_type == "gaussian":

        return gaussian_noise(client_update)

    elif attack_type == "random":

        return random_weights(client_update)

    elif attack_type == "zero":

        return zero_weights(client_update)

    else:

        return client_update