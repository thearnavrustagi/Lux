class Module(object):
    def __init__ (self):
        raise NotImplementedError("init method needs to be implemented for a module")

    def __call__ (self, *arg):
        self.forward(*args)

    def forward(self,*args):
        raise NotImplementedError("the forward method should be implemented for a module")
