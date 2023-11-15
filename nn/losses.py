from torch import nn

class CrossEntropyLoss (object):
    def __init__ (self):
        pass

    def __call__ (self, y_pred, y_target):
        loss = y_pred * (nn.softmax(y_target).log())
        return loss
