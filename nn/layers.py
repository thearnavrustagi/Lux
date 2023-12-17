import ToyTorch
import numpy as np

class Layer(object):
    def __init__ (self):
        raise NotImplementedError("init function should be defined for a Layer Class")

    def __call__ (self, *args):
        self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError("forward function should be defined for a Layer Class")

class Linear(Layer):
    def __init__ (
            self,
            in_features:int,
            out_features:int,
            bias:bool=False):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = ToyTorch.initialise_weights((in_features, out_features))
        self.B = ToyTorch.initialise_weights((out_features,))

    def forward(self,x):
        x = np.matmul(self.W,x)
        x = x + self.B

        return x

