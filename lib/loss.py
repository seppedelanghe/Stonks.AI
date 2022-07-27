import torch.nn as nn

class LSTMLoss(nn.Module):
    def __init__(self):
        super(LSTMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        
    def forward(self, pred, target):
        return self.mse(pred, target)