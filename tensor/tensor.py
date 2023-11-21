from typing_extensions import Callable

import numpy as np
from numpy import ndarray

from ToyTorch.autograd import *


class Tensor(ndarray):
    def __new__(
        subtype,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        is_leaf: bool = True,
        req_grad: bool = False,
        grad: ndarray = None,
        grad_fn: Callable = None,
        *args,
        **kwargs,
    ) -> None:
        obj = super().__new__(subtype, shape, dtype, buffer, offset, strides, order)

        obj.is_leaf = is_leaf
        obj.req_grad = req_grad
        obj.grad = grad
        obj.grad_fn = grad_fn

        obj.version = 1

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        attributes = ["is_leaf", "req_grad", "grad", "grad_fn", "version"]

        for attr in attributes:
            exec(f"self.{attr}=getattr(obj,'{attr}',None)")
            print(exec(f"self.{attr}"))

    @staticmethod
    def tensor(
        collection,
        is_leaf: bool = True,
        req_grad: bool = False,
        grad: ndarray = None,
        grad_fn: Callable = None,
        *args,
        **kwargs,
    ):
        obj = np.array(collection, *args, *kwargs).view(Tensor)
        obj.is_leaf = is_leaf
        obj.req_grad = req_grad
        obj.grad = grad
        obj.grad_fn = grad_fn

        obj.version = 1

        return obj

    def exp(self):
        y = call_function(Exp, (self,))
        return y


if __name__ == "__main__":
    t = Tensor(shape=(3, 4))
    print(t)
    print("shape = ", t.shape)
    print("is_leaf = ", t.is_leaf)
    print("req_grad = ", t.req_grad)
    print("grad = ", t.grad)
    print("grad_fn = ", t.grad_fn)
