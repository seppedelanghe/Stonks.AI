import torch.nn as nn

class ConvLSTMLoss(nn.Module):
    def __init__(self):
        super(ConvLSTMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.huber = nn.HuberLoss(reduction='sum', delta=1)

    def forward(self, pred, target):
        loss = self.huber(pred[:, :-1], target[:, :-1]) * 10e3 + self.huber(pred[:, -1:], target[:, -1:]) * 10e2
        
        return loss