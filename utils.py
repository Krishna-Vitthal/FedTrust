import torch

from config import DEVICE


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