import numpy as np


class Exp(object):
    def __init__(self):
        pass

    def __call__(self, X):
        return np.exp(X)


class Reciprocal(object):
    def __init__(self):
        pass

    def __call__(self, X):
        return np.reciprocal(X)


class Tanh(object):
    def __init__(self):
        pass

    def __call__(self, X):
        return np.tanh(X)


class Log(object):
    def __init__(self):
        pass

    def __call__(self, X):
        return np.log(X)


class Geometric(object):
    def __init__(self):
        pass

    def __call__(self, X, a):
        return np.power(X, a)


class Sigmoid(object):
    def __init__(self):
        pass

    def __call__(self, X):
        return 1.0 / (1.0 + np.exp(-X))


class ReLU(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x - x * (x < 0)


if __name__ == "__main__":
    pass
