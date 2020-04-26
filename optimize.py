from utils import get_device, load_model
from plotting import generate_lineplot

import logging
import torch
import torch.nn.functional as F


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=25):
    logging.info("Starting training...")
    device = get_device()
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0
        num_batches = 0

        # Iterate over data
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward input through model
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculate gradients, and update weights
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validate for each epoch
        val_loss = val_model(model, criterion, val_loader)

        # Append losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logging.info("Epoch {}, train_loss: {}, val_loss: {}".format(epoch, train_loss, val_loss))

    # Save model to file
    torch.save(model.state_dict(), "./model.pth")

    logging.info("Generating plot to loss.png")
    generate_lineplot([train_losses, val_losses])


def val_model(model, criterion, val_loader):
    model.eval()
    device = get_device()
    with torch.no_grad():
        val_loss = 0
        num_batches = 0

        # Iterate over data
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward input through model
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculate gradients, and update weights

            val_loss += loss.item()
            num_batches += 1

        val_loss /= num_batches
    return val_loss


def test_model(test_loader):
    """
    Test the best model saved on the test dataset
    :param test_loader: Dataset containing test images
    :return:
    """
    device = get_device()
    model = load_model("./model.pth")
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward through model
            outputs = model(inputs)

            # Take softmax and convert to predicted labels
            preds = F.softmax(outputs, dim=1)
            preds = torch.argmax(preds, 1)
            #logging.info("{}, {}".format(preds, labels))
            correct += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

    logging.info("Test Accuracy: {}%".format(round(100 * (correct / total), 3)))
