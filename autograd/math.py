import numpy as np

from ToyTorch.autograd.function import Function


class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        y = np.exp(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_for_backward
        return y * grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx, x, a):
        y = np.multiply(x, a)
        ctx.save_for_backward(x, a)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_for_backward
        return grad_output * a


if __name__ == "__main__":
    pass
