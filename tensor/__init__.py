from .tensor import *

def zeros(self, shape):
    obj = np.zeros(shape).view(Tensor)
    obj.is_leaf = is_leaf
    obj.req_grad = req_grad
    obj.grad = grad
    obj.grad_fn = grad_fn

    obj.version = 1
    return obj
