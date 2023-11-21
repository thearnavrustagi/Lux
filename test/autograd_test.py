import ToyTorch

if __name__ == "__main__":
    print("hello world")
    tensor = ToyTorch.Tensor.tensor([0, 1, 2, 3, 4])
    tensor = tensor.exp()
    print(tensor, type(tensor))
    out = tensor.grad_fn()
    print(out)
