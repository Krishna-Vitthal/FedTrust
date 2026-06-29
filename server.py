import copy

from model import CNN

from aggregation import (
    fedavg,
    fedtrust
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

        # ---------------------------------------
        # FedTrust
        # ---------------------------------------

        else:

            trust_scores = self.trust_manager.process_round(
                client_updates
            )

            new_weights = fedtrust(

                client_updates,

                trust_scores

            )

        self.global_model.load_state_dict(
            new_weights
        )

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