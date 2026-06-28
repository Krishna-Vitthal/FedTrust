import copy

from model import CNN
from aggregation import fedavg
from config import DEVICE


class Server:

    def __init__(self):

        # Global Model
        self.global_model = CNN().to(DEVICE)

    # ----------------------------------

    def get_weights(self):

        return copy.deepcopy(
            self.global_model.state_dict()
        )

    # ----------------------------------

    def aggregate(self, client_updates):

        new_weights = fedavg(client_updates)

        self.global_model.load_state_dict(
            new_weights
        )