class Sigmoid(object):
    def __init__ (self):
        pass
    
    def __call__ (self, x):
        return 1/(1+(-x).exp())

class TanH(object):
    def __init__ (self):
        self.__sigmoid__ = Sigmoid()

    def __call__ (self, x):
        return 2 * self.__sigmoid__(2*x) - 1


class RelU (object):
    def __init__ (self):
        pass

    def __call__ (self,x):
        return x - x * (x < 0)
