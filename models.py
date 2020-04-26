import torch.nn as nn
import torchvision as tv


class RockPaperScissorsClassifier(nn.Module):

    def __init__(self):
        super(RockPaperScissorsClassifier, self).__init__()

        # Load pre-trained model and freeze weights
        self.model = tv.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        # Change last layer to fit our classsification task
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 4)

    def forward(self, images):
        return self.model(images)
