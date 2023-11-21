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
):  # , gradient=None, retain_graph=None, create_graph=False, inputs=None):
    gradient = 1
    queue = [(tensor, 1)]
    while len(queue):
        print(queue)
        t_queue = []
        for in_tensor, arg in queue:
            if not in_tensor.grad_fn:
                continue
            out, next_tensors = in_tensor.grad_fn(arg)
            in_tensor.grad = out
            next_tensors = list(map(lambda x: (x, out), next_tensors))
            t_queue += next_tensors
        queue = t_queue
