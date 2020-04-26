import torchvision as tv
import torch


def get_dataloader(root, batch_size, shuffle=True, transform=None):
    """
    Return a dataloader of all images in root
    :param root: str - path to folder with all the images in their corresponding class folder
    :param batch_size: int
    :param shuffle: boolean
    :param transform: torchvision.transforms.Compose
    :return: torch.utils.data.dataloader.DataLoader
    """
    data = tv.datasets.ImageFolder(root, transform=transform)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                                       num_workers=4)
