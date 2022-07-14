import torch
import torch.nn as nn
from lib.utils import prod

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(CNNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

    def outshape(self, w, h):
        return int((h + 2 * self.padding - 1 * (self.kernel_size - 1) - 1) / self.stride + 1), int((w + 2 * self.padding - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)


class ConvLSTM(nn.Module):
    def __init__(self, time_d: int = 10, n_inputs: int = 6, n_outputs: int = 5):
        super(ConvLSTM, self).__init__()
        self.time_d = time_d
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.dropout = 0.1
        self.n_filters = 32

        self.cnn = CNNBlock(
                    1,        # channels
                    out_channels=self.n_filters,  # filters
                    kernel_size=2,   # kernel size
                    stride=2,        # stride
                    padding=3,       # padding
            )
        self.cnn_bn = nn.BatchNorm2d(self.n_filters)

        self.lstm = nn.LSTM(time_d, n_inputs, 2)
        self.lstm_bn = nn.BatchNorm1d(n_inputs)

        self.mid_neurons = (self.n_inputs ** 2) + (self.n_filters * prod(self.cnn.outshape(self.n_inputs, self.time_d)))
        self.final = self._create_output_layers()

    def _create_output_layers(self):
        return nn.Sequential(
            nn.Linear(self.mid_neurons, (self.time_d * self.n_inputs) ** 2),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),

            nn.Linear((self.time_d * self.n_inputs) ** 2, self.time_d * self.n_inputs),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),

            nn.Linear(self.time_d * self.n_inputs, self.n_outputs)
        )

    def forward(self, x):
        xa, (h_n, c_n) = self.lstm(x.reshape(-1, self.n_inputs, self.time_d))
        xa = self.lstm_bn(xa)

        xb = self.cnn(x.reshape(-1, 1, self.time_d, self.n_inputs))
        xb = self.cnn_bn(xb)

        xa = xa.reshape(-1, self.n_inputs * self.n_inputs)
        xb = xb.reshape(-1, self.n_filters * prod(self.cnn.outshape(self.n_inputs, self.time_d)))

        x = torch.hstack((xa, xb))

        # forward through output layers and return
        return self.final(x)