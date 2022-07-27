import torch
import torch.nn as nn

class StockModule(nn.Module):
    def __init__(self, gamma: float, outputs: int = 1, lrelu: float = 0.1, use_bias: bool = True):
        super(StockModule, self).__init__()

        self.gamma = gamma
        self.linear = nn.Linear(4, outputs, bias=use_bias)
        self.bnorm = nn.BatchNorm1d(outputs)
        self.lrelu = nn.LeakyReLU(lrelu)

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == 4, "StockModule only accepts inputs that have 4 values in their final dim: open, close, low, high. Got {}".format(x.shape[-1])

        x = self.linear(x)
        x = self.bnorm(x)
        x = self.lrelu(x)

        return x * self.gamma

class StockTimeModule(nn.Module):
    def __init__(self, time: int, outputs: int, gamma: float = 0.9, inner_outputs: int = 2,  device: str = 'cpu'):
        super(StockTimeModule, self).__init__()

        self.device = device
        self.time = time
        self.outputs = outputs
        self.gamma = gamma
        self.inner_outputs = inner_outputs

        self.layers = self._make_layers()
        self.inner = self._make_inner()

    def _make_layers(self):
        return [
            StockModule(self.gamma ** (self.time - t), self.inner_outputs).to(self.device)
            for t in range(0, self.time)
        ]

    def _make_inner(self):
        return nn.Sequential(
            nn.Linear(self.time * self.inner_outputs, self.time),
            nn.BatchNorm1d(self.time),

            nn.Linear(self.time, self.outputs),
            nn.BatchNorm1d(self.outputs),
            
            nn.LeakyReLU(0.1)
        )

    def forward(self, x: torch.Tensor):
        x = [self.layers[i](x[:, i]) for i in range(x.shape[1])]
        x = torch.stack(x).permute((1, 0, 2)).flatten(1, 2)
        return self.inner(x)