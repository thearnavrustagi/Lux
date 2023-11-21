from .function import *
from .functions import *
from .graph import *


def call_function(function, arguments):
    ctx = FunctionCtx()
    y = function.forward(ctx, *arguments)
    y.grad_fn = lambda: function.backward(ctx, y)
    y.version += 1
    return y
