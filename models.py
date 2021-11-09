## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

from typing import List


class Net(nn.Module):
    N_KERNEL = 3
    N_POOL = 2
    P_DROP = 0.2
    CONV_LAYERS = (1, 32, 64, 128, 256)
    FC_LAYERS = (1024, 1024, 136)
    N_INPUT = 224
    N_OUTPUT = 68

    def __init__(self):
        super(Net, self).__init__()
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # ACM - completed

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)

        # Init Conv_out param
        self.conv_out = self.N_INPUT

        # Build
        self.pool = nn.MaxPool2d(self.N_POOL, self.N_POOL)
        self.drop = nn.Dropout(self.P_DROP)
        self.act = nn.ReLU()
        self.final_act = nn.Tanh()

        self.conv = self._build_conv()
        self.fc = self._build_fc()

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and
        # other layers (such as dropout or batch normalization) to avoid
        # overfitting

    def _build_conv(self) -> nn.Sequential:
        """
        Construct the convolutional layer end-to-end

        Architecture expects 4 convolutional layers MaxPool2s in between. If
        any of these parameters change, update this function and the supporting
        function _update_conv_out.
        """
        layers = []
        for i in range(1, len(self.CONV_LAYERS)):
            channels = self.CONV_LAYERS[i - 1], self.CONV_LAYERS[i]
            layers.append(
                nn.Sequential(
                    nn.Conv2d(*channels, self.N_KERNEL),
                    self.act,
                    self.pool,
                    self.drop,
                )
            )
            self._update_conv_out()
        return nn.Sequential(*layers)

    def _update_conv_out(self) -> None:
        """Update conv_out parameter with a step in conv_stack"""
        # Calculate conv_out from conv step
        self.conv_out = (self.conv_out - self.N_KERNEL) // 1 + (1 + 2 * 0)
        # Calculate conv_out from maxpool step
        self.conv_out = self.conv_out // self.N_POOL

    def _get_fc_in(self) -> int:
        """Gather dimension of first FC layer"""
        return self.conv_out ** 2 * self.CONV_LAYERS[-1]

    def _build_fc(self) -> nn.Sequential:
        layers = []
        fc_layers = [self._get_fc_in(), *self.FC_LAYERS]
        # add first and middle layers
        for i in range(1, len(fc_layers) - 1):
            channels = fc_layers[i - 1], fc_layers[i]
            layers.append(nn.Sequential(nn.Linear(*channels), self.act, self.drop))
        # Add final Layer
        layers.append(
            nn.Sequential(
                nn.Linear(*fc_layers[-2:]),
                self.final_act,
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        # ACM - completed
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        conv_out = self.conv(x)
        fc_in = conv_out.view(conv_out.size(0), -1)
        flat_out = self.fc(fc_in)
        out = flat_out.view(flat_out.size(0), self.N_OUTPUT, -1)

        # a modified x, having gone through all the layers of your model, should be returned
        return out


# optimizer = torch.optim.Adam(lr=LR, momentum=MO)
# criterion = nn.MSELoss()
