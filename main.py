from utils import init_logger
from camera import WebCamera

if __name__ == '__main__':
    init_logger()
    cam = WebCamera(None)
    cam.capture_training_data()
