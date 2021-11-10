"""Define the neural net for facial keypoint detection"""
## TODOne: define the convolutional neural network architecture
from torch import nn

# can use the below import should you choose to initialize the weights of your
# Net


class Net(nn.Module):
    """Nerual Network for facial Keypoint detection"""

    N_KERNEL = 3
    N_POOL = 2
    P_DROP = 0.2
    CONV_LAYERS = (1, 32, 64, 128, 256)
    DENSE_LAYERS = (1024, 1024, 136)
    N_INPUT = 224
    # N_OUTPUT = 68

    def __init__(self):
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

        # Init Conv_out param
        self.conv_out = self.N_INPUT

        # Build
        self.pool = nn.MaxPool2d(self.N_POOL, self.N_POOL)
        # self.drop = nn.Dropout(self.P_DROP)
        self.act = nn.ReLU()
        # self.final_act = nn.Threshold(1, 1) # nn.Tanh()

        self.conv = self._build_conv()
        self.dense = self._build_dense()
        # self.init_weights()

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
                    # self.drop,
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

    def _get_dense_in(self) -> int:
        """Gather dimension of first Dense layer"""
        return self.conv_out ** 2 * self.CONV_LAYERS[-1]

    def _build_dense(self) -> nn.Sequential:
        layers = []
        dense_layers = [self._get_dense_in(), *self.DENSE_LAYERS]
        # add first and middle layers
        for i in range(1, len(dense_layers) - 1):
            channels = dense_layers[i - 1], dense_layers[i]
            layers.append(
                nn.Sequential(
                    nn.Linear(*channels),
                    self.act,
                    # self.drop,
                )
            )
        # Add final Layer
        layers.append(
            nn.Sequential(
                nn.Linear(*dense_layers[-2:]),
                # self.final_act,
            )
        )
        return nn.Sequential(*layers)

    # def init_weights(self) -> None:
    #     """Initialize the weights of the NN"""
    #     nn.init.

    def forward(self, x_in):
        """
        Run the neural net forward.
        Called when class is called through __call__
        """
        # TODOne: Define the feedforward behavior of this model
        # ACM - completed
        # x is the input image and, as an example, here you may choose to
        # include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        conv_out = self.conv(x_in)
        dense_in = conv_out.view(conv_out.size(0), -1)
        out = self.dense(dense_in)
        # out = flat_out.view(flat_out.size(0), self.N_OUTPUT, -1)

        # a modified x, having gone through all the layers of your model, should
        # be returned
        return out
