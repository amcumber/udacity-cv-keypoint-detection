"""Define the neural net for facial keypoint detection"""
## TODOne: define the convolutional neural network architecture
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential

# can use the below import should you choose to initialize the weights of your
# Net


class Net(nn.Module):
    """Nerual Network for facial Keypoint detection"""

    ACT = nn.ReLU()

    def __init__(
        self,
        n_input: int = 224,
        n_output: int = 136,
        dense_size: int = 1024,
    ):
        """Constuctor"""
        super().__init__()
        # TODOne: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscal
        #     image as input
        # 2. It ends with a linear layer that represents the keypoints
        #     it's suggested that you make this last layer output 136 values, 2
        #     for each of the 68 keypoint (x, y) pairs
        # ACM - completed

        # As an example, you've been given a convolutional layer, which you may
        # (but don't have to) change: 1 input image channel (grayscale), 32
        # output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)

        # inital architecture https://arxiv.org/pdf/1710.00977.pdf
        conv_out = n_input

        # Build
        self.conv = nn.ModuleList()
        width_in = 1
        p_drop = 0.1
        for i, width_out in zip(range(4), (32, 64, 128, 256)):
            k = 5 if i % 2 else 3
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(width_in, width_out, k),
                    self.ACT,
                    nn.MaxPool2d(2, 2),
                    # nn.Dropout(p_drop),
                )
            )
            p_drop += 0.1
            width_in = width_out
            conv_out = self._get_conv_side(conv_out, k=k) // 2

        self.dense = nn.ModuleList()
        self.dense.append(
            nn.Sequential(
                nn.Linear(conv_out ** 2 * width_out, dense_size),
                self.ACT,
                # nn.Dropout(p_drop),
            )
        )
        p_drop += 0.1

        self.dense.append(
            nn.Sequential(
                nn.Linear(dense_size, dense_size),
                self.ACT,
                # nn.Dropout(p_drop),
            )
        )
        p_drop += 0.1

        self.dense.append(
            nn.Sequential(
                nn.Linear(dense_size, dense_size),
                self.ACT,
                # nn.Dropout(p_drop),
            )
        )
        p_drop += 0.1

        self.dense.append(nn.Sequential(nn.Linear(dense_size, n_output)))

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and
        # other layers (such as dropout or batch normalization) to avoid
        # overfitting

    @staticmethod
    def _get_conv_side(conv_in, k=3, s=1, p=0) -> None:
        """Get convolutional side after a pass"""
        return (conv_in - k) // s + (1 + 2 * p)

    def forward(self, x):
        """
        Run the neural net forward.
        Called when class is called through __call__
        """
        # TODOne: Define the feedforward behavior of this model
        # ACM - completed
        # x is the input image and, as an example, here you may choose to
        # include a pool/conv step:
        for layer in self.conv:
            x = layer(x)

        x = x.view(x.size(0), -1)

        for layer in self.dense:
            x = layer(x)

        out = x
        # out = flat_out.view(flat_out.size(0), self.N_OUTPUT, -1)

        # a modified x, having gone through all the layers of your model, should
        # be returned
        return out
