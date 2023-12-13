from ToyTorch import optim
import ToyTorch

def optimize(x_init=4.,y_init=3.):
    x = ToyTorch.Tensor.tensor([x_init, y_init])
    res = []
    adam = optim.Adam([x], alpha=0.1)
    vals = [x.copy()]
    for _ in range(100):
        y = x**2
        y.backward()
        print(x.grad)
        adam.step()
        vals.append(x.copy())
    return vals

if __name__ == "__main__":
    optimize()
    x = ToyTorch.Tensor.tensor([1.,2.])
    b = ToyTorch.Tensor.tensor([3.,4.])
    print("Initial Values of x,b : ",x,b)
    adam = optim.Adam([x,b], alpha=0.1)
    x.grad = ToyTorch.Tensor.tensor(5.,5.)
    b.grad = ToyTorch.Tensor.tensor(5.,5.)
    adam.step()
    print("Final values of x,b : ",x,b)
