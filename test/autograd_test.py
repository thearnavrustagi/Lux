import ToyTorch

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Autograd")
    a = ToyTorch.Tensor.tensor([2, 3])
    x = ToyTorch.Tensor.tensor([0, 1])
    print(f"input tensor : x={x},a={a}")
    c = x * a
    tensor = c.exp()
    print(f"modified tensor after exp(c);c=a*b : {tensor}")
    tensor.backward()
    print(f"gradient of tensor (x) after exponential : {x.grad}")
    print(f"gradient of tensor (c) after exponential : {c.grad}")
    print(f"gradient of tensor (tensor) after exponential : {tensor.grad}")
    print("=" * 80)
