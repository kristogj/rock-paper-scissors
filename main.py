from utils import init_logger, load_config, preview_images, load_model
from camera import capture_training_data, predict_live
from dataloader import get_dataloader
from models import RockPaperScissorsClassifier
from optimize import train_model, test_model

import torch.nn as nn
from torchvision import transforms
import torch.optim as optim

if __name__ == '__main__':
    init_logger()
    config = load_config("config.yaml")

    # Remove comment to collect training data
    # capture_training_data()

    transformer = transforms.Compose([
        transforms.Resize(size=128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    # Load dataset
    train_loader, class_to_idx = get_dataloader("./data/train", batch_size=config["batch_size"], shuffle=True,
                                                transform=transformer)
    val_loader, _ = get_dataloader("./data/val", batch_size=1, shuffle=False, transform=transformer)
    test_loader, _ = get_dataloader("./data/test", batch_size=1, shuffle=False, transform=transformer)

    # Preview images
    #preview_images(train_loader, list(class_to_idx.keys()))

    # Initialize model
    rps = RockPaperScissorsClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rps.model.fc.parameters())

    # Train model
    #train_model(rps, criterion, optimizer, train_loader, val_loader, epochs=10)

    # Test model
    #test_model(test_loader)

    # Load model from file
    model = load_model("./model.pth")
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # Make live predictions from camera
    predict_live(model, transformer, idx_to_class)
