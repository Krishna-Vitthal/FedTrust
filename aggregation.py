import copy

from config import *


# =====================================================
# Adaptive Trust Filtering
# =====================================================

def filter_clients(client_updates, trust_scores):

    if not USE_TRUST_FILTER:

        return client_updates

    trusts = [

        trust_scores[c["client_id"]]

        for c in client_updates

    ]

    if TRUST_FILTER_MODE == "fixed":

        threshold = FIXED_TRUST_THRESHOLD

    else:

        mean = sum(trusts) / len(trusts)

        variance = sum(

            (t - mean) ** 2

            for t in trusts

        ) / len(trusts)

        std = variance ** 0.5

        threshold = max(

            MIN_TRUST,

            mean - STD_FACTOR * std

        )

    filtered = []

    for client in client_updates:

        trust = trust_scores[

            client["client_id"]

        ]

        if trust >= threshold:

            filtered.append(client)

    # Never allow empty aggregation
    if len(filtered) == 0:

        filtered = client_updates

    return filtered


# =====================================================
# Standard FedAvg
# =====================================================

def fedavg(client_updates):

    total_samples = sum(

        client["num_samples"]

        for client in client_updates

    )

    avg_weights = copy.deepcopy(

        client_updates[0]["weights"]

    )

    # -------------------------------------

    first_weight = (

        client_updates[0]["num_samples"]

        / total_samples

    )

    for key in avg_weights:

        avg_weights[key] *= first_weight

    # -------------------------------------

    for client in client_updates[1:]:

        weight = (

            client["num_samples"]

            / total_samples

        )

        for key in avg_weights:

            avg_weights[key] += (

                client["weights"][key] * weight

            )

    return avg_weights


# =====================================================
# FedTrust Aggregation
# =====================================================

def fedtrust(client_updates, trust_scores):

    client_updates = filter_clients(

        client_updates,

        trust_scores

    )

    # -------------------------------------
    # Compute Trust-Weighted Sample Count
    # -------------------------------------

    total_weight = 0.0

    for client in client_updates:

        client_id = client["client_id"]

        total_weight += (

            trust_scores[client_id]

            * client["num_samples"]

        )

    # -------------------------------------

    aggregated = copy.deepcopy(

        client_updates[0]["weights"]

    )

    first_client = client_updates[0]

    first_weight = (

        trust_scores[first_client["client_id"]]

        * first_client["num_samples"]

    ) / total_weight

    for key in aggregated:

        aggregated[key] *= first_weight

    # -------------------------------------

    for client in client_updates[1:]:

        client_id = client["client_id"]

        weight = (

            trust_scores[client_id]

            * client["num_samples"]

        ) / total_weight

        for key in aggregated:

            aggregated[key] += (

                client["weights"][key]

                * weight

            )

    return aggregated