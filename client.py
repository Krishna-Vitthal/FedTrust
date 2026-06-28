import copy

import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN
from config import *


class Client:

    def __init__(self, client_id, train_loader):

        self.client_id = client_id

        self.train_loader = train_loader

        self.model = CNN().to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()

    # ----------------------------

    def set_weights(self, global_weights):

        self.model.load_state_dict(
            copy.deepcopy(global_weights)
        )

    # ----------------------------

    def train(self):

        self.model.train()

        # Fresh optimizer every communication round
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        total_loss = 0

        for epoch in range(LOCAL_EPOCHS):

            for images, labels in self.train_loader:

                images = images.to(DEVICE)

                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

        return {

            "weights": copy.deepcopy(
                self.model.state_dict()
            ),

            "num_samples": len(
                self.train_loader.dataset
            ),

            "loss": total_loss

        }