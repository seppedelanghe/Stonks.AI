import torch
import torch.nn as nn

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
    def __init__(self, time_d: int, n_inputs: int, n_outputs: int):
        super(ConvLSTM, self).__init__()
        self.time_d = time_d
        self.small_time_d = 5
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.dropout = 0.1
        self.n_filters = (64, 128, 32)

        self.cnn = self._make_conv()
        self.cnn_bn = nn.BatchNorm1d(self.n_inputs * self.n_outputs * self.n_filters[-1])

        # large lstm for all data
        self.lstm = nn.LSTM(n_inputs, time_d, 2)
        self.lstm_bn = nn.BatchNorm1d(time_d)

        # small lstm for most recent data
        self.lstm_s = nn.LSTM(n_inputs, self.small_time_d, 1)
        self.lstm_s_bn = nn.BatchNorm1d(self.small_time_d)

        self.mid_neurons = (self.time_d ** 2) + (self.small_time_d ** 2) + (self.n_inputs * self.n_outputs * self.n_filters[-1])
        self.final = self._create_output_layers()

    def _make_conv(self):
        layers = [
            CNNBlock(
                    1,        # channels
                    out_channels=self.n_filters[0],  # filters
                    kernel_size=2,   # kernel size
                    stride=2,        # stride
                    padding=3,       # padding
            ),
            CNNBlock(
                    self.n_filters[0],     # channels
                    out_channels=self.n_filters[1],  # filters
                    kernel_size=2,   # kernel size
                    stride=2,        # stride
                    padding=3,       # padding
            ),
            nn.MaxPool2d(2, 2),
            CNNBlock(
                    self.n_filters[1],        # channels
                    out_channels=self.n_filters[2],  # filters
                    kernel_size=2,   # kernel size
                    stride=2,        # stride
                    padding=3,       # padding
            ),
            nn.Flatten(),
        ]
        return nn.Sequential(*layers)

    def _create_output_layers(self):
        return nn.Sequential(
            nn.Linear(self.mid_neurons, (self.time_d * self.n_inputs) ** 2),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),

            nn.Linear((self.time_d * self.n_inputs) ** 2, self.time_d * self.n_inputs),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(0.1),

            nn.Linear(self.time_d * self.n_inputs, self.n_outputs),
        )

    def forward(self, x):
        xa, (h_n, c_n) = self.lstm(x)
        xa: torch.Tensor = self.lstm_bn(xa)

        xb = self.cnn(x.reshape(-1, 1, self.time_d, self.n_inputs))
        xb: torch.Tensor = self.cnn_bn(xb)

        xc, (h_n_s, c_n_s) = self.lstm_s(x[:, :self.small_time_d])
        xc: torch.Tensor = self.lstm_s_bn(xc)

        xa = xa.reshape(-1, self.time_d ** 2)
        xb = xb.reshape(-1, self.n_inputs * self.n_outputs * self.n_filters[-1])
        xc = xc.reshape(-1, self.small_time_d ** 2)

        x = torch.hstack((xa, xc, xb))

        # forward through output layers and return
        return self.final(x)