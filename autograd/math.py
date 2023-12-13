import numpy as np

from ToyTorch.autograd.function import Function


class Add(Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(x, a)
        return x + a

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_for_backward
        return grad_output * 1


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
        ctx.save_for_backward(x, a)
        return np.multiply(x, a)

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_for_backward
        return grad_output * a


class Reciprocal(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.reciprocal(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_for_backward
        return grad_output * np.negative(np.reciprocal(np.square(x)))


class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.tanh(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_for_backward
        return grad_output * (1 - np.tanh(x) ** 2)


class Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_for_backward
        return grad_output * np.reciprocal(x)


class Geometric(Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(x, a)
        return np.power(x, a)

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_for_backward
        return grad_output * np.multiply(np.power(x, a - 1), a)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        y = 1.0 / (1.0 + np.exp(-x))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_for_backward
        return grad_output * (y * (1 - y))


class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x - x * (x < 0)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_for_backward
        return grad_output * (1 - 1 * (x < 0))


if __name__ == "__main__":
    pass
