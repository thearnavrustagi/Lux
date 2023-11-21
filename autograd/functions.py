import numpy as np

from ToyTorch.autograd.function import Function


class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        y = np.exp(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, y):
        (x,) = ctx.saved_for_backward
        return y * x


if __name__ == "__main__":
    pass
