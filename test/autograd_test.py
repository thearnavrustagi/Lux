import ToyTorch
import torch

if __name__ == "__main__":
    print("=" * 80)
    print("TOYTORCH")
    print("Testing Autograd")
    a = ToyTorch.Tensor.tensor([2, 3])
    x = ToyTorch.Tensor.tensor([4, 1])
    print(f"input tensor : x={x},a={a}")
    c = x * a
    tensor = c.exp()
    print(f"modified tensor after exp(c);c=a*b : {tensor}")
    tensor.backward()
    print(f"gradient of tensor (x) after exponential : {x.grad}")
    print(f"gradient of tensor (c) after exponential : {c.grad}")
    print(f"gradient of tensor (y) after exponential : {tensor.grad}")

    a,x = torch.tensor([2.,3.],requires_grad=True), torch.tensor([4.,1.], requires_grad=True)
    c = x*a
    y = c.exp()

    c = torch.tensor([8.,3.], requires_grad=True)
    z = c.exp()

    y.backward(gradient=torch.tensor([1.,1.]))
    z.backward(gradient=torch.tensor([1.,1.]))
    print("="*80)
    print("PYTORCH")
    print(f"expected gradient of tensor (x) after exponential : {x.grad}")
    print(f"expected gradient of tensor (c) after exponential : {c.grad}")

    print("=" * 80)
