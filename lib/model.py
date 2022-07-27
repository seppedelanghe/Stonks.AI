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


class LSTMModel(nn.Module):
    def __init__(self, inputs: int, hidden: int, layers: int, outputs: int, device: str):
        super(LSTMModel, self).__init__()

        self.hidden = hidden
        self.outputs = outputs
        self.layers = layers
        self.inputs = inputs
        self.device = device

        self.lstm = nn.LSTM(inputs, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, outputs)

    def forward(self, x: torch.Tensor):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layers, x.size(0), self.hidden).to(self.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layers, x.size(0), self.hidden).to(self.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> batch_size, hidden, batch_size
        # out[:, -1, :] --> batch_size, batch_size --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :])
        return out

# class ConvLSTM(nn.Module):
#     def __init__(self, time_d: int, n_inputs: int, n_outputs: int, device: str = 'cuda'):
#         super(ConvLSTM, self).__init__()
#         self.time_d = time_d
#         self.n_outputs = n_outputs
#         self.n_inputs = n_inputs
#         self.dropout = 0.1
#         self.n_filters = (32, 32, 32)

#         # cnn for patterns
#         self.cnn = self._make_conv()

#         # lstm for patterns
#         self.lstm = nn.LSTM(n_inputs, time_d, 2)
#         self.lstm_bn = nn.BatchNorm1d(self.time_d ** 2)

#         # custom stock data neuron for short term
#         self.stm = StockTimeModule(self.time_d, self.n_outputs, 0.95, 2, device)

#         self.mid_neurons = self.time_d ** 2 + self.n_outputs + 768
#         self.final = self._create_output_layers()

#     def _make_conv(self):
#         layers = [
#             CNNBlock(
#                     1,        # channels
#                     out_channels=self.n_filters[0],  # filters
#                     kernel_size=2,   # kernel size
#                     stride=2,        # stride
#                     padding=3,       # padding
#             ),
#             CNNBlock(
#                     self.n_filters[0],     # channels
#                     out_channels=self.n_filters[1],  # filters
#                     kernel_size=2,   # kernel size
#                     stride=2,        # stride
#                     padding=3,       # padding
#             ),
#             nn.MaxPool2d(2, 2),
#             CNNBlock(
#                     self.n_filters[1],        # channels
#                     out_channels=self.n_filters[2],  # filters
#                     kernel_size=2,   # kernel size
#                     stride=2,        # stride
#                     padding=3,       # padding
#             ), # => outshape is 7x3x32 => 896
#             nn.Flatten(),
#             nn.BatchNorm1d(768)
#         ]
#         return nn.Sequential(*layers)

#     def _create_output_layers(self):
#         return nn.Sequential(
#             nn.Linear(self.mid_neurons, (self.time_d * self.n_inputs) ** 2),
#             nn.Dropout(self.dropout),
#             nn.LeakyReLU(0.1),

#             nn.Linear((self.time_d * self.n_inputs) ** 2, self.time_d * self.n_inputs),
#             nn.Dropout(self.dropout),
#             nn.LeakyReLU(0.1),

#             nn.Linear(self.time_d * self.n_inputs, self.n_outputs),
#         )

#     def forward(self, x):
#         xa, (h_n, c_n) = self.lstm(x)
#         xa = xa.reshape(-1, self.time_d ** 2)
#         xa = self.lstm_bn(xa)

#         xb: torch.Tensor = self.cnn(x.reshape(-1, 1, self.time_d, self.n_inputs))
#         xc: torch.Tensor = self.stm(x[:, :, :4])

#         # stack all outputs
#         x = torch.hstack((xa, xc, xb))

#         # forward through output layers and return
#         return self.final(x)