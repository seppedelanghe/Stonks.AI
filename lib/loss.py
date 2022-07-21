import torch
import torch.nn as nn

class ConvLSTMLoss(nn.Module):
    def __init__(self):
        super(ConvLSTMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.huber = nn.HuberLoss(reduction='sum', delta=1)

    def forward(self, pred, target):
        # high_low_target = target[:, 2] - target[:, 0]
        # high_low_pred = pred[:, 2] - pred[:, 0]
        # high_low_loss = torch.sum(torch.sqrt((high_low_pred - high_low_target) ** 2))

        # open_close_target = target[:, 1] - target[:, 3]
        # open_close_pred = pred[:, 1] - pred[:, 3]
        # open_close_loss = torch.sum(torch.sqrt((open_close_target - open_close_pred) ** 2))

        huber_loss = self.huber(pred, target) 
        # loss = huber_loss + high_low_loss + open_close_loss

        return huber_loss