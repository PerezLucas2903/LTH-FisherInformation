import torch
import torch.nn as nn
import torchvision.models as models

class ConvModelMNIST(nn.Module):
    def __init__(self):
        super(ConvModelMNIST, self).__init__()
        self.net = nn.Sequential( # input [batch_size, 1, 28, 28]
            nn.Conv2d(1, 4, 3),   # [batch_size, 4, 26, 26]
            nn.ReLU(),            #
            nn.MaxPool2d(2),      # [batch_size, 4, 13, 13]
            nn.Conv2d(4, 16, 4),   # [batch_size, 16, 10, 10]
            nn.ReLU(),            #  
            nn.MaxPool2d(2),      # [batch_size, 16, 5, 5]
            nn.Flatten(),         # [batch_size, 400]
            nn.Linear(400,10)     # output [batch_size, 10]
        )
    def forward(self, X):
        return self.net(X)
    

class ConvModelEMNIST(nn.Module):
    def __init__(self, n_classes=10):
        super(ConvModelEMNIST, self).__init__()
        self.net = nn.Sequential( # input [batch_size, 1, 28, 28]
            nn.Conv2d(1, 4, 3),   # [batch_size, 4, 26, 26]
            nn.ReLU(),            #
            nn.MaxPool2d(2),      # [batch_size, 4, 13, 13]
            nn.Conv2d(4, 16, 4),   # [batch_size, 16, 10, 10]
            nn.ReLU(),            #  
            nn.MaxPool2d(4),      # [batch_size, 16, 5, 5]
            nn.Flatten(),         # [batch_size, 400]
            nn.Linear(64,n_classes)     # output [batch_size, n_classes]
        )
    def forward(self, X):
        return self.net(X)
    
def mobilenet(num_classes: int = 10) -> nn.Module:
    model = models.mobilenet_v2(num_classes=num_classes)
    return model

def resnet18(num_classes: int = 10, imagenet=False) -> nn.Module:
    model = models.resnet18(num_classes=num_classes)
    if not imagenet:
        # Replace the 7x7 stride-2 conv + maxpool with a 3x3 stride-1 conv and no pool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model

def resnet50(num_classes: int = 10, imagenet=False) -> nn.Module:
    model = models.resnet50(num_classes=num_classes)
    if not imagenet:
        # Replace the 7x7 stride-2 conv + maxpool with a 3x3 stride-1 conv and no pool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model

def wide_resnet(num_classes: int = 10, imagenet=False) -> nn.Module:
    model = models.wide_resnet50_2(num_classes=num_classes)
    if not imagenet:
        # Replace the 7x7 stride-2 conv + maxpool with a 3x3 stride-1 conv and no pool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model

def densenet121(num_classes: int = 10) -> nn.Module:
    model = models.densenet121(num_classes=num_classes)
    return model

def convnext_tiny(num_classes: int = 10) -> nn.Module:
    model = models.convnext_tiny(num_classes=num_classes)
    return model
