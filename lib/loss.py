import torch.nn as nn

class ConvLSTMLoss(nn.Module):
    def __init__(self):
        super(ConvLSTMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        return loss