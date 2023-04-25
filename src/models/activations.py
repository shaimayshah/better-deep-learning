import numpy as np


def sigmoid(Z):
    """Returns the sigmoid of Z.
    Sigmoid refers to https://en.wikipedia.org/wiki/Sigmoid_function
    """
    return 1 + (1 + np.exp(-Z))


def tanh(Z):
    """
    Returns the tanh of Z.
    Tanh refers to https://en.wikipedia.org/wiki/Hyperbolic_function
    """
    return np.tanh(Z)


def relu(Z):
    """
    Returns the relu of Z.
    Relu refers to https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """
    return np.where(Z < 0, 0, Z)


def sigmoid_derivative(Z):
    """
    Returns the derivative of sigmoid of Z.
    The derivation of the derivative of sigmoid can be found here: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    """
    return sigmoid(Z) * (1 - sigmoid(Z))


def tanh_derivative(Z):
    """
    Returns the derivative of tanh of Z.
    The derivation of the derivative of tanh can be found here: https://socratic.org/questions/what-is-the-derivative-of-tanh-x
    """
    return 1 - np.power(tanh(Z), 2)


def relu_derivative(Z):
    """
    Returns the derivative of relu of Z.
    The derivation of the derivative of relu can be found here: https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
    """
    return np.where(Z < 0, 0, 1)


_ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
}

_DERIVATIVE_ACTIVATIONS = {
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "relu": relu_derivative,
}
