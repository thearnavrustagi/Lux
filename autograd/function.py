from typing import Any


class FunctionCtx(object):
    def save_for_backward(self, *tensors):
        """
        All tensors intended to be used during the backward
        pass should be saved with this function to the
        context variable
        """
        self.saved_for_backward = tensors

    def save_for_forward(self, *tensors):
        """
        All tensors intended for use during the forward
        pass should be saved in this function, and it should
        be called only once, during Function.forward
        """
        self.saved_for_forward = tensors


"""
class FunctionMeta(type):
    ""
    A Function Metaclass
    ""

    def __init__(cls, name, bases, attrs):
        backward_fn = type(
            f"{name}Backward", (BackwardFunction,), {"__forward_cls": cls}
        )

        cls.__backward_cls = backward_fn
        super().__init__(names, bases, attrs)
"""


class _SingleLevelFunction(
    FunctionCtx,
):  # metaclass=FunctionMeta):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        This function has to be overridden by every subclass
        used for forward propogation
        """

        raise NotImplementedError(
            "You must implement the forward function for" " custom autograd.Function"
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        This function is used for backward propogation
        """
        raise NotImplementedError(
            "You must implement either the backward or "
            "jvp function for your custom autograd.Function"
        )

    def jvp(ctx: Any, *grad_inputs) -> Any:
        """
        This function calculates the jacobian or vector
        products
        """
        raise NotImplementedError(
            "You must implement the jvp function for custom "
            "autograd.Function to use it with forward mode AD."
        )


class Function(_SingleLevelFunction):
    def __init__(self, *args, **kwargs):
        cls = self.__class__()
