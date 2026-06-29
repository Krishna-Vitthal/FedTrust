import copy

import torch
import torch.nn.functional as F

from collections import defaultdict

from config import (
    DEVICE,
    PEER_SIMILARITY_THRESHOLD
)


# =====================================================
# Evaluation
# =====================================================

def evaluate(model, test_loader, criterion):

    model.eval()

    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            batch_loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            loss += batch_loss.item() * batch_size

            _, predicted = torch.max(outputs, 1)

            total += batch_size

            correct += (predicted == labels).sum().item()

    if total == 0:

        return 0.0, 0.0

    accuracy = 100 * correct / total

    avg_loss = loss / total

    return accuracy, avg_loss


# =====================================================
# Flatten Model Update
# =====================================================

def flatten_weights(weights):

    vectors = []

    for tensor in weights.values():

        vectors.append(
            tensor.view(-1)
        )

    return torch.cat(vectors)


# =====================================================
# Cosine Similarity
# =====================================================

def cosine_similarity(weights1, weights2):

    v1 = flatten_weights(weights1)

    v2 = flatten_weights(weights2)

    similarity = F.cosine_similarity(
        v1.unsqueeze(0),
        v2.unsqueeze(0)
    )

    return similarity.item()


# =====================================================
# Gradient Norm
# =====================================================

def gradient_norm(weights):

    vector = flatten_weights(weights)

    return torch.norm(vector, p=2).item()


# =====================================================
# Histogram Similarity
# =====================================================

def histogram_similarity(hist1, hist2):

    h1 = torch.tensor(hist1, dtype=torch.float32)

    h2 = torch.tensor(hist2, dtype=torch.float32)

    similarity = F.cosine_similarity(
        h1.unsqueeze(0),
        h2.unsqueeze(0)
    )

    return similarity.item()


# =====================================================
# Build Peer Groups
# =====================================================

def build_peer_groups(client_updates):

    peer_groups = defaultdict(list)

    assigned = set()

    group_id = 0

    for i, client in enumerate(client_updates):

        if i in assigned:

            continue

        assigned.add(i)

        peer_groups[group_id].append(i)

        hist_i = client["label_histogram"]

        for j in range(i + 1, len(client_updates)):

            if j in assigned:

                continue

            hist_j = client_updates[j]["label_histogram"]

            sim = histogram_similarity(
                hist_i,
                hist_j
            )

            if sim >= PEER_SIMILARITY_THRESHOLD:

                assigned.add(j)

                peer_groups[group_id].append(j)

        group_id += 1

    return peer_groups


# =====================================================
# Peer Average Update
# =====================================================

def peer_average_update(client_updates, peer_indices, exclude_index=None, fallback_weights=None):

    avg = {}

    peer_list = [
        index
        for index in peer_indices
        if index != exclude_index
    ]

    if not peer_list:

        if fallback_weights is None:

            return global_average_update(client_updates)

        return copy.deepcopy(fallback_weights)

    first = client_updates[
        peer_list[0]
    ]["weights"]

    for key in first:

        avg[key] = torch.zeros_like(
            first[key]
        )

    for idx in peer_list:

        weights = client_updates[idx]["weights"]

        for key in weights:

            avg[key] += weights[key]

    for key in avg:

        avg[key] /= len(peer_list)

    return avg


# =====================================================
# Global Average Update
# =====================================================

def global_average_update(client_updates):

    avg = {}

    first = client_updates[0]["weights"]

    for key in first:

        avg[key] = torch.zeros_like(
            first[key]
        )

    for client in client_updates:

        weights = client["weights"]

        for key in weights:

            avg[key] += weights[key]

    for key in avg:

        avg[key] /= len(client_updates)

    return avg


# =====================================================
# Gradient Norm Score
# =====================================================

def compute_norm_scores(client_updates):

    norms = []

    for client in client_updates:

        norms.append(
            gradient_norm(client["weights"])
        )

    median = torch.median(
        torch.tensor(norms)
    ).item()

    scores = []

    for norm in norms:

        score = torch.exp(
            torch.tensor(
                -abs(norm - median) /
                (median + 1e-8)
            )
        ).item()

        scores.append(score)

    return scores


# =====================================================
# Normalize Scores
# =====================================================

def normalize_scores(scores):

    minimum = min(scores)

    maximum = max(scores)

    if maximum - minimum < 1e-8:

        return [1.0] * len(scores)

    normalized = []

    for score in scores:

        normalized.append(

            (score - minimum) /
            (maximum - minimum)

        )

    return normalized


# =====================================================
# Alignment Score
# =====================================================

def compute_alignment_scores(client_updates):

    global_update = global_average_update(
        client_updates
    )

    scores = []

    for client in client_updates:

        similarity = cosine_similarity(

            client["weights"],

            global_update

        )

        scores.append(

            (similarity + 1) / 2

        )

    return scores