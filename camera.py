import cv2
import logging
import os
from PIL import Image
import torch
import torch.nn.functional as F

# Rectangle position
x1, y1, x2, y2 = 800, 100, 1200, 500


def capture_training_data():
    """
    Collect images that are going to be used as the training data for the model
    :return: None
    """
    _class = "None"
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture training data")
    img_counter = 0
    logging.info("Initialized WebCamera")
    while True:
        # Read frame from camera and show image in viewer
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        # Update how many images has been captured
        cv2.putText(frame, "{} Images Captured: {}".format(_class, img_counter), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0))

        # Instructions on how to use camera and save to correct folder
        cv2.putText(frame, "SPACE: capture, ESC: close, r: rock, s: scissors, p: paper, n: none", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Square area to capture images from
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.imshow("Capture training data", frame)

        if not ret:
            break
        k = cv2.waitKey(1)

        # Change save folder for captured images
        if k == ord("r"):
            _class = "Rock"
            img_counter = 0
        elif k == ord("p"):
            _class = "Paper"
            img_counter = 0
        elif k == ord("s"):
            _class = "Scissors"
            img_counter = 0
        elif k == ord("n"):
            _class = "None"
            img_counter = 0

        if k % 256 == 27:
            # ESC pressed - close window
            logging.info("Escaped pressed, closing down camera...")
            break
        elif k % 256 == 32:
            # SPACE pressed - save captured image to file
            folder = "./data/{}".format(_class.lower())
            if not os.path.isdir(folder):
                os.mkdir(folder)
            img_name = folder + "/{}.png".format(img_counter)
            cv2.imwrite(img_name, frame[y1:y2, x1:x2])
            logging.info("Saved {}".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()


def predict_live(model, transformer, idx_to_class):
    """
    Predict rock, paper, scissors or none inside rectangle
    :param model: RockPaperScissorsClassifier
    :param transformer: torchvision.transforms.Compose
    :param idx_to_class: dict - index to class name
    :return:
    """
    _class = "None"
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Predict Live")
    logging.info("Predicting Live on WebCamera")
    while True:
        # Read frame from camera and show image in viewer
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Square area to capture images from
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Make prediction if what is showing inside the rectangle
        img = frame[y1:y2, x1:x2]
        img = transformer(Image.fromarray(img))
        img = img[None, :, :, :]
        pred = F.softmax(model(img), dim=1)
        pred = torch.argmax(pred, 1).item()
        _class = idx_to_class[pred]

        # Update how many images has been captured
        cv2.putText(frame, "Prediction: {}".format(_class), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        cv2.imshow("Predict Live", frame)

        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed - close window
            logging.info("Escaped pressed, closing down camera...")
            break

    cam.release()
    cv2.destroyAllWindows()
