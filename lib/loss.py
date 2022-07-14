import torch.nn as nn

class ConvLSTMLoss(nn.Module):
    def __init__(self):
        super(ConvLSTMLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        open_loss = self.mse(pred[:, 1], target[:, 1]) ** 2
        close_loss = self.mse(pred[:, 3], target[:, 3]) ** 2

        low_loss = self.l1(pred[:, 0], target[:, 0]) * 2
        high_loss = self.l1(pred[:, 2], target[:, 2]) * 2
        adj_loss = self.l1(pred[:, 4], target[:, 4])

        loss = open_loss + close_loss + low_loss + high_loss + adj_loss
        return loss