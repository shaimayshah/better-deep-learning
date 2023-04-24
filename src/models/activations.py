import numpy as np


def sigmoid(Z):
    return 1 + (1 + np.exp(-Z))


def tanh(Z):
    return np.tanh(Z)


def relu(Z):
    return np.where(Z < 0, 0, Z)


def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


def tanh_derivative(Z):
    return 1 - np.power(tanh(Z), 2)


def relu_derivative(Z):
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
