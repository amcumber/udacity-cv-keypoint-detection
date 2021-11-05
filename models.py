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
    FC_LAYERS = (512, 512)
    N_OUTPUT = 64
    N_INPUT = 28

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
        self.convs = nn.Sequential(*self.build_conv_layers())
        self.fcs = nn.Sequential(*self.build_fc_layers())
        self.final_output = self.build_final_layer()
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def _get_conv_stack(self, channel_in, channel_out) -> nn.Sequential:
        """
        Return a convolutional stack with preconfigured parameters and updates
        expected conv_out parameter

        Parameters
        ----------
        channel_in : int
            number of channels inputs
        channel_out : int
            number of channels outputs
        """
        stack =  nn.Sequential(
            nn.Conv2d(channel_in, channel_out, self.N_KERNEL),
            nn.ReLU(),
            nn.MaxPool2d(self.N_POOL, self.N_POOL),
            nn.Dropout(self.P_DROP)
        )
        # Calculate conv_out from conv step
        self._update_conv_out()
        return stack

    def _update_conv_out(self) -> None:
        """Update conv_out parameter with a step in conv_stack"""
        # Calculate conv_out from conv step
        self.conv_out = (self.conv_out - self.N_KERNEL) //1 + (1 + 2 * 0)
        # Calculate conv_out from maxpool step
        self.conv_out = self.conv_out // self.N_POOL

    def build_conv_layers(self) -> List[nn.Sequential]:
        """Build the convolutional layers based on get_conv_stack method"""
        prev_layer = None
        layers = []
        for layer in self.CONV_LAYERS:
            if prev_layer is not None:
                layers.append(
                    self._get_conv_stack(prev_layer, layer)
                )
            prev_layer = layer
        return layers
        
    def _get_fc_stack(self, channel_in, channel_out) -> nn.Sequential:
        """
        Return a FC stack with preconfigured parameters
        Parameters
        ----------
        channel_in : int
            number of channels inputs
        channel_out : int
            number of channels outputs
        """
        return nn.Sequential(
            nn.Dropout(self.P_DROP),
            nn.Linear(channel_in, channel_out),
            nn.ReLU(),
        )

    def build_fc_layers(self) -> List[nn.Sequential]:
        """Build the Fully Connected Layers"""
        first_fc = nn.Sequential(
            nn.Linear(self.conv_out, self.FC_LAYERS[0]),
            nn.ReLU,
        )
        prev_layer = None
        layers = [first_fc]
        for layer in self.FC_LAYERS:
            if prev_layer is not None:
                layers.append(
                    self._get_fc_stack(prev_layer, layer)
                )
            prev_layer = layer
        return layers

    def build_final_layer(self) -> List[nn.Sequential]:
        """Build the final layer of the network"""
        return nn.Sequential(
            nn.Linear(self.FC_LAYERS[-1], self.N_OUTPUT),
        )

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        # ACM - completed
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        conv_out = self.convs(x)
        fc_in = conv_out.view(conv_out.size(0), -1)
        fc_out = self.fcs(fc_in)
        out = self.final_output(fc_out)

        # a modified x, having gone through all the layers of your model, should be returned
        return out
