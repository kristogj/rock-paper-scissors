from utils import init_logger, load_config
from camera import WebCamera
from dataloader import get_dataloader
from torchvision import transforms

if __name__ == '__main__':
    init_logger()
    config = load_config("config.yaml")

    cam = WebCamera(None)

    # Remove comment to collect training data
    # cam.capture_training_data()

    transformer = transforms.Compose([
        transforms.Resize(size=28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    # Load dataset
    dataloader_train = get_dataloader("./data/train", batch_size=config["batch_size"], shuffle=True,
                                      transform=transformer)
    dataloader_test = get_dataloader("./data/test", batch_size=1, shuffle=False, transform=transformer)
