from torch import nn

class MSELoss(object):
    def __init__(self):
        pass

    def __call__(self, y_pred, y_target):
        loss = (y_pred - y_target)**2
        return loss
