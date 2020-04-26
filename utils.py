import logging
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import torch


def init_logger():
    """
    Initialize logger settings
    :return: None
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("app.log", mode="w"),
            logging.StreamHandler()
        ])


def load_config(path):
    """
    Load the configuration from task_2_table.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def preview_images(dataloader, class_names):
    # Get a batch of training data
    inputs, classes = next(iter(dataloader))

    # Make a grid from batch
    out = tv.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
