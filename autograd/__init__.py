from .function import *
from .math import *


def call_function(function, arguments):
    ctx = FunctionCtx()
    y = function.forward(ctx, *arguments)

    next_tensors = arguments

    y.grad_fn = lambda y: (function.backward(ctx, y), next_tensors)
    y.version += 1
    return y


def backward(
    tensor,
    gradient=1
):  
    defer = True
    queue = [(tensor, 1)]
    while len(queue):
        t_queue = []
        for in_tensor, prev_grad in queue:
            if not in_tensor.grad_fn:
                in_tensor.grad = prev_grad
                continue
            grad, next_tensors = in_tensor.grad_fn(prev_grad)
            in_tensor.grad = prev_grad
            next_tensors = list(map(lambda x: (x, grad), next_tensors))
            t_queue += next_tensors
            old_grad = grad
        queue = t_queue

