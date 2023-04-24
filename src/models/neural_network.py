from typing import List, Dict

import numpy as np

from src.models.activations import _ACTIVATIONS, _DERIVATIVE_ACTIVATIONS


class NeuralNetwork:
    DEFAULT_HYPERPARAMS = {
        "learning_rate": 0.05,
    }

    def __init__(
        self,
        layer_sizes: List[int] = None,
        activations: List[str] = None,
        hyper_params: Dict = None,
    ):
        self.layer_sizes = [12288, 7, 1] if layer_sizes is None else layer_sizes
        self.activations = (
            ["relu", "sigmoid"] if activations is None else activations
        )
        self.cache = {}
        self.hyper_params = (
            self.DEFAULT_HYPERPARAMS if hyper_params is None else hyper_params
        )
        self.parameters = {}
        self.grads = {}
        self.n_layers = len(self.layer_sizes) - 1

    def _initialize_parameters(self, seed=2023):
        for layer in range(1, self.n_layers + 1):
            np.random.seed(seed)
            curr_layer = self.layer_sizes[layer]
            prev_layer = self.layer_sizes[layer - 1]
            self.parameters[f"W{layer}"] = (
                np.random.randn(curr_layer, prev_layer) * 0.01
            )
            self.parameters[f"b{layer}"] = np.zeros((curr_layer, 1))

    def _propagate_forward(self, X):
        self.cache["A0"] = X
        for ind, layer in enumerate(range(1, self.n_layers + 1)):
            A_prev = self.cache[f"A{layer - 1}"]

            activation = _ACTIVATIONS[self.activations[ind]]

            W = self.parameters[f"W{layer}"]
            b = self.parameters[f"b{layer}"]

            Z = np.dot(W, A_prev) + b
            A = activation(Z)

            self.cache[f"Z{layer}"] = Z
            self.cache[f"A{layer}"] = A

    @staticmethod
    def _compute_cost(AL, Y):
        m = Y.shape[1]
        cost = -(np.dot(Y, np.log(AL.T)) + np.dot(1 - Y, np.log(1 - AL.T))) / m

        return np.squeeze(cost)

    def _propagate_backward(self, Y):
        m = Y.shape[1]

        AL = self.cache[f"A{self.n_layers}"]
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        activation = _DERIVATIVE_ACTIVATIONS[self.activations[-1]]
        dZ = dAL * activation(self.cache[f"Z{self.n_layers}"])

        self.grads[f"dW{self.n_layers}"] = (
            np.dot(dZ, self.cache[f"A{self.n_layers-1}"].T) / m
        )
        self.grads[f"db{self.n_layers}"] = np.sum(dZ, axis=1, keepdims=True) / m

        for layer in reversed(range(1, self.n_layers)):
            dA_prev = np.dot(self.parameters[f"W{layer+1}"].T, dZ)
            activation = _DERIVATIVE_ACTIVATIONS[self.activations[layer]]
            dZ = dA_prev * activation(self.cache[f"Z{layer}"])
            self.grads[f"dW{layer}"] = (
                np.dot(dZ, self.cache[f"A{layer-1}"].T) / m
            )
            self.grads[f"db{layer}"] = np.sum(dZ, axis=1, keepdims=True) / m

    def _update_params(self):
        learning_rate = self.hyper_params["learning_rate"]
        for layer in range(1, self.n_layers):
            self.parameters[f"W{layer}"] -= (
                learning_rate * self.grads[f"dW{layer}"]
            )
            self.parameters[f"b{layer}"] -= (
                learning_rate * self.grads[f"db{layer}"]
            )

    def fit(self, X, Y, num_iterations=500, verbose=False):
        self._initialize_parameters()
        for iteration in range(num_iterations):
            self._propagate_forward(X)
            self._propagate_backward(Y)
            self._update_params()
            cost = self._compute_cost(self.cache[f"A{self.n_layers}"], Y)
            if verbose:
                if num_iterations % 10 == 0:
                    print(f"Cost: {cost}")

    def predict(self, X):
        self._propagate_forward(X)
        AL = self.cache[f"A{self.n_layers}"]
        return (AL > 0.5).astype(int)
