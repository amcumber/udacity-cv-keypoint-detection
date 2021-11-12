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

    def __init__(
        self,
        n_input: int = 224,
        n_output: int = 136,
        p_drop_init: float = 0.1,
        max_p_drop: float = 0.25,
        n_conv: int = 5,
        n_fc: int = 3,
        act_fun: callable = nn.ReLU(),
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
        conv_structure = (32, 64, 128, 256, 512)
        hidden_structure = (1024, 512)

        # Build
        self.act_f = act_fun

        width_in = width_out = 1
        p_drop = p_drop_init
        update_p_drop = self._update_p_drop(max_p_drop=max_p_drop)

        self.conv = nn.ModuleList()
        for i, width_out in zip(range(n_conv), conv_structure[:n_conv]):
            # k = 5 if i % 2 else 3
            k = 3 if i < n_conv - 1 else 1
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(width_in, width_out, k),
                    self.act_f,
                    nn.MaxPool2d(2, 2),
                    # nn.Dropout(p_drop),
                )
            )
            p_drop = update_p_drop(p_drop)
            width_in = width_out
            conv_out = self._get_conv_side(conv_out, k=k) // 2

        self.dense = nn.ModuleList()
        # n = 1
        self.dense.append(
            nn.Sequential(
                nn.Linear(conv_out ** 2 * width_out, hidden_structure[0]),
                self.act_f,
                nn.Dropout(p_drop),
            )
        )
        p_drop = update_p_drop(p_drop)

        dense_in = hidden_structure[0]
        for i in range(n_fc - 2):
            dense_out = hidden_structure[i + 1]
            self.dense.append(
                nn.Sequential(
                    nn.Linear(dense_in, dense_out),
                    self.act_f,
                    nn.Dropout(p_drop),
                )
            )
            p_drop = update_p_drop(p_drop)
            dense_in = dense_out

        # n = -1
        self.dense.append(
            nn.Sequential(
                nn.Linear(
                    hidden_structure[-1],
                    n_output,
                )
            )
        )

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and
        # other layers (such as dropout or batch normalization) to avoid
        # overfitting

    @staticmethod
    def _get_conv_side(conv_in, k=3, s=1, p=0) -> None:
        """Get convolutional side after a pass"""
        return (conv_in - k) // s + (1 + 2 * p)

    @staticmethod
    def _update_p_drop(max_p_drop=0.5) -> float:
        """Update dropout probability by increasing by 10% up to max"""

        def wrapper(p_drop):
            p_drop += 0.1
            return p_drop if p_drop < max_p_drop else max_p_drop

        return wrapper

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
