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

        # -----------------------------------
        # Compute Label Histogram Once
        # -----------------------------------

        self.label_histogram = self.compute_label_histogram()

    # =====================================================
    # Label Histogram
    # =====================================================

    def compute_label_histogram(self):

        histogram = torch.zeros(10)

        total = 0

        for _, labels in self.train_loader:

            for label in labels:

                histogram[label.item()] += 1

                total += 1

        if total > 0:

            histogram /= total

        return histogram.tolist()

    # =====================================================
    # Load Global Model
    # =====================================================

    def set_weights(self, global_weights):

        self.model.load_state_dict(

            copy.deepcopy(global_weights)

        )

    # =====================================================
    # Local Training
    # =====================================================

    def train(self):

        self.model.train()

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

        # -----------------------------------
        # Send Local Update
        # -----------------------------------

        return {

            "client_id": self.client_id,

            "weights": copy.deepcopy(

                self.model.state_dict()

            ),

            "num_samples": len(

                self.train_loader.dataset

            ),

            "loss": total_loss,

            "label_histogram": self.label_histogram

        }