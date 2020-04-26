from utils import get_device
from plotting import generate_lineplot

import logging
import torch


def train_model(model, criterion, optimizer, dataloader, epochs=25):
    logging.info("Starting training...")
    device = get_device()
    model.train()
    train_losses = []
    for epoch in range(1, epochs + 1):

        train_loss = 0
        num_batches = 0

        # Iterate over data
        for i, (inputs, labels) in enumerate(dataloader):
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

        logging.info("Epoch {}, loss: {}".format(epoch, train_loss))

        # Append losses
        train_losses.append(train_loss)

    logging.info("Generating plot to loss.png")
    generate_lineplot(train_losses)
