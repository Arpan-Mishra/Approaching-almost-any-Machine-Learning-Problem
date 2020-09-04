# AlexNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        # convolution part
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 96,
        kernel_size = 11, stride=4, padding=0)

        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256,
        kernel_size = 5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384,
        kernel_size = 3, stride = 1, padding = 1)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256,
        kernel_size=3, stride=1, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # dense part
        self.fc1 = nn.Linear(9216, 4096)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(4096, 1000)


    def forward(self, image):
        bs, c, h, w = image.size() # (bs, 3, 227, 227)
        x = F.relu(self.conv1(image)) # (bs, 96, 55, 55)
        x = self.pool1(x) # (bs, 96, 27, 27)

        x = F.relu(self.conv2(x)) # (bs, 256, 27, 27)
        x = self.pool2(x) # (bs, 256, 13, 13)

        x = F.relu(self.conv3(x)) # (bs, 384, 13, 13)
        
        x = F.relu(self.conv4(x)) # (bs, 256, 13, 13)
        x = self.pool3(x) # (bs, 256, 6, 6)

        x = x.view(bs, -1) # (bs, 9216)

        x = F.relu(self.fc1(x)) # (bs, 4096)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x)) # (bs, 4096)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))

        x = torch.softmax(x, axis = 1)

        return x

        


