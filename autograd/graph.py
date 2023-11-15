def backward(tensor, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    gradient = tensor.grad_fn()
