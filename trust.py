import copy
import torch

from config import *

from utils import (
    cosine_similarity,
    build_peer_groups,
    peer_average_update,
    global_average_update,
    compute_alignment_scores as utils_compute_alignment_scores,
    compute_norm_scores as utils_compute_norm_scores
)


class TrustManager:

    # =====================================================
    # Constructor
    # =====================================================

    def __init__(self, num_clients):

        self.num_clients = num_clients

        # -------------------------------
        # Trust Scores
        # -------------------------------

        self.trust_scores = {

            client_id: INITIAL_TRUST

            for client_id in range(num_clients)

        }

        self.recovery_counter = {
            client_id: 0
            for client_id in range(num_clients)
        }

        # -------------------------------
        # Previous Updates
        # -------------------------------

        self.previous_updates = {}

        # -------------------------------
        # Behavior Cache
        # -------------------------------

        self.similarity_scores = {}

        self.consistency_scores = {}

        self.alignment_scores = {}

        self.global_alignment_scores = {}

        self.norm_scores = {}

        self.behavior_scores = {}

        self.trust_history = []

        self.peer_groups = None

    # =====================================================
    # Peer Group Construction
    # =====================================================

    def create_peer_groups(self, client_updates):

        self.peer_groups = build_peer_groups(

            client_updates

        )

        return self.peer_groups

    # =====================================================
    # Distribution-Aware Similarity
    # =====================================================

    def compute_similarity_scores(self, client_updates):

        scores = {}

        global_update = global_average_update(

            client_updates

        )

        if self.peer_groups is None:

            self.create_peer_groups(

                client_updates

            )

        # ---------------------------------------------

        for group in self.peer_groups.values():

            for index in group:

                peer_average = peer_average_update(

                    client_updates,

                    group,

                    exclude_index=index,

                    fallback_weights=global_update
                )

                client = client_updates[index]

                similarity = cosine_similarity(

                    client["weights"],

                    peer_average

                )

                similarity = (similarity + 1) / 2

                scores[

                    client["client_id"]

                ] = similarity

        self.similarity_scores = scores

        return scores

    # =====================================================
    # Historical Consistency
    # =====================================================

    def compute_consistency_scores(self, client_updates):

        scores = {}

        for client in client_updates:

            client_id = client["client_id"]

            current = client["weights"]

            # -------------------------------------

            if client_id not in self.previous_updates:

                scores[client_id] = 0.5

            else:

                previous = self.previous_updates[

                    client_id

                ]

                consistency = cosine_similarity(

                    current,

                    previous

                )

                consistency = (

                    consistency + 1

                ) / 2

                scores[client_id] = consistency

            # -------------------------------------

            self.previous_updates[

                client_id

            ] = copy.deepcopy(current)

        self.consistency_scores = scores

        return scores

    # =====================================================
    # Getters
    # =====================================================

    def get_similarity_scores(self):

        return self.similarity_scores

    def get_consistency_scores(self):

        return self.consistency_scores

    def get_peer_groups(self):

        return self.peer_groups

    def get_trust_scores(self):

        return self.trust_scores

    def get_trust_history(self):

        return self.trust_history

    # =====================================================
    # Global Alignment Score (Ai)
    # =====================================================

    def compute_alignment_scores(self, client_updates):

        alignment = utils_compute_alignment_scores(
            client_updates
        )

        scores = {}

        for idx, client in enumerate(client_updates):

            scores[
                client["client_id"]
            ] = alignment[idx]

        self.alignment_scores = scores

        self.global_alignment_scores = scores

        return scores


    # =====================================================
    # Gradient Norm Score (Ni)
    # =====================================================

    def compute_norm_scores(self, client_updates):

        norm_scores = utils_compute_norm_scores(
            client_updates
        )

        scores = {}

        for idx, client in enumerate(client_updates):

            scores[
                client["client_id"]
            ] = norm_scores[idx]

        self.norm_scores = scores

        return scores


    # =====================================================
    # Behavior Score
    # =====================================================

    def compute_behavior_scores(self):

        scores = {}

        for client_id in self.trust_scores.keys():

            S = self.similarity_scores.get(client_id, 0.5)

            C = self.consistency_scores.get(client_id, 0.5)

            A = self.alignment_scores.get(client_id, 0.5)

            N = self.norm_scores.get(client_id, 0.5)

            behavior = (

                SIMILARITY_WEIGHT * S +

                CONSISTENCY_WEIGHT * C +

                ALIGNMENT_WEIGHT * A +

                NORM_WEIGHT * N

            )

            behavior = max(
                0.0,
                min(1.0, behavior)
            )

            scores[client_id] = behavior

        self.behavior_scores = scores

        return scores


    # =====================================================
    # EMA Trust Update
    # =====================================================

    def update_trust(self):

        for client_id in self.trust_scores.keys():

            old_trust = self.trust_scores[client_id]

            behavior = self.behavior_scores[
                client_id
            ]

            new_trust = (

                ALPHA * old_trust +

                (1 - ALPHA) * behavior

            )

            if behavior >= RECOVERY_THRESHOLD:
                self.recovery_counter[client_id] += 1
            else:
                self.recovery_counter[client_id] = 0

            if self.recovery_counter[client_id] >= RECOVERY_ROUNDS:
                new_trust += RECOVERY_STEP
                self.recovery_counter[client_id] = 0

            new_trust = max(

                MIN_TRUST,

                min(MAX_TRUST, new_trust)

            )

            self.trust_scores[client_id] = new_trust

        self.trust_history.append(

            copy.deepcopy(self.trust_scores)

        )


    # =====================================================
    # Complete Trust Pipeline
    # =====================================================

    def process_round(self, client_updates):

        self.create_peer_groups(
            client_updates
        )

        self.compute_similarity_scores(
            client_updates
        )

        self.compute_consistency_scores(
            client_updates
        )

        self.compute_alignment_scores(
            client_updates
        )

        self.compute_norm_scores(
            client_updates
        )

        self.compute_behavior_scores()

        self.update_trust()

        return self.trust_scores


    # =====================================================
    # Additional Getters
    # =====================================================

    def get_behavior_scores(self):

        return self.behavior_scores


    def get_alignment_scores(self):

        return self.alignment_scores


    def get_global_alignment_scores(self):

        return self.global_alignment_scores


    def get_norm_scores(self):

        return self.norm_scores
