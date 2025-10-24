import torch
import torch.nn as nn




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
