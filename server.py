import copy

from model import CNN

from aggregation import (
    fedavg,
    fedtrust,
    filter_clients
)

from trust import TrustManager

from config import (
    DEVICE,
    NUM_CLIENTS,
    USE_TRUST
)


class Server:

    # =====================================================
    # Constructor
    # =====================================================

    def __init__(self):

        self.global_model = CNN().to(DEVICE)

        self.trust_manager = TrustManager(
            NUM_CLIENTS
        )

        self.last_aggregation_info = {}

    # =====================================================
    # Global Weights
    # =====================================================

    def get_weights(self):

        return copy.deepcopy(
            self.global_model.state_dict()
        )

    # =====================================================
    # Global Model
    # =====================================================

    def get_model(self):

        return self.global_model

    # =====================================================
    # Aggregation
    # =====================================================

    def aggregate(self, client_updates):

        # ---------------------------------------
        # Standard FedAvg
        # ---------------------------------------

        if not USE_TRUST:

            new_weights = fedavg(
                client_updates
            )

            self.last_aggregation_info = {
                "mode": "fedavg",
                "filtered_client_ids": [],
                "filtered_out_client_ids": [],
                "filtered_count": 0,
                "filtered_out_count": 0,
                "total_clients": len(client_updates),
            }

        # ---------------------------------------
        # FedTrust
        # ---------------------------------------

        else:

            trust_scores = self.trust_manager.process_round(
                client_updates
            )

            filtered_updates = filter_clients(
                client_updates,
                trust_scores
            )
            kept_client_ids = {
                client["client_id"]
                for client in filtered_updates
            }

            new_weights = fedtrust(

                client_updates,

                trust_scores

            )

            self.last_aggregation_info = {
                "mode": "fedtrust",
                "filtered_client_ids": [
                    client["client_id"]
                    for client in filtered_updates
                ],
                "filtered_out_client_ids": [
                    client["client_id"]
                    for client in client_updates
                    if client["client_id"] not in kept_client_ids
                ],
                "filtered_count": len(filtered_updates),
                "filtered_out_count": len(client_updates) - len(filtered_updates),
                "total_clients": len(client_updates),
                "trust_scores": trust_scores,
            }

        self.global_model.load_state_dict(
            new_weights
        )

        return self.last_aggregation_info

    # =====================================================
    # Trust Information
    # =====================================================

    def get_trust_scores(self):

        return self.trust_manager.get_trust_scores()

    def get_trust_history(self):

        return self.trust_manager.get_trust_history()

    def get_behavior_scores(self):

        return self.trust_manager.get_behavior_scores()

    def get_similarity_scores(self):

        return self.trust_manager.get_similarity_scores()

    def get_consistency_scores(self):

        return self.trust_manager.get_consistency_scores()

    def get_alignment_scores(self):

        return self.trust_manager.get_alignment_scores()

    def get_norm_scores(self):

        return self.trust_manager.get_norm_scores()

    def get_peer_groups(self):

        return self.trust_manager.get_peer_groups()

    def get_last_aggregation_info(self):

        return self.last_aggregation_info