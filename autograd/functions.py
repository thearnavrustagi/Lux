from function import Function


class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = np.exp(i)
        ctx.save_for_backward(i)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_for_backward
        return grad_output * result


if __name__ == "__main__":
    pass
