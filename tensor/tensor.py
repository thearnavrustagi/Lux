from typing_extensions import Callable
from numpy import ndarray


class Tensor(ndarray):
    def __init__(
        self,
        is_leaf: bool = True,
        req_grad: bool = False,
        grad: ndarray = None,
        grad_fn: Callable = None,
        **kwargs,
    ) -> None:
        self.is_leaf = is_leaf
        self.req_grad = req_grad
        self.grad = grad
        self.grad_fn = grad_fn


if __name__ == "__main__":
    t = Tensor(shape=(3, 4))
    print(t)
    print("shape = ", t.shape)
    print("is_leaf = ", t.is_leaf)
    print("req_grad = ", t.req_grad)
    print("grad = ", t.grad)
    print("grad_fn = ", t.grad_fn)
