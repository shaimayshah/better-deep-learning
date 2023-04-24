from typing import List, Dict

import numpy as np

from src.models.activations import _ACTIVATIONS, _DERIVATIVE_ACTIVATIONS


class NeuralNetwork:
    """A neural network class for binary classification.

    Attributes:
        layer_sizes (List[int]): The number of nodes in each layer of the
        neural network including the input layer.
        activations (List[str]): The activation function for each layer of
        the neural network excluding the input layer. len(activations) should
        be 1 less than len(layer_sizes)
        hyper_params (Dict): Hyperparameters for the neural network.
        cache (Dict): Dictionary to store intermediate results during forward
        propagation.
        parameters (Dict): Dictionary to store the weights and biases of each
        layer.
        grads (Dict): Dictionary to store the gradients of each weight and
        bias.
        n_layers (int): The number of layers in the neural network.

    Methods:
        _initialize_parameters(seed): Initializes the weights and biases of
        each layer randomly.
        _propagate_forward(X): Computes the forward propagation for the given
        input.
        _compute_cost(AL, Y): Computes the binary cross-entropy cost
        function.
        _propagate_backward(Y): Computes the backward propagation to compute
        gradients.
        _update_params(): Updates the weights and biases of each layer.
        fit(X, Y, num_iterations, verbose): Trains the neural network using
        gradient descent.
        predict(X): Predicts the binary output for the given input.
    """

    DEFAULT_HYPERPARAMS = {
        "learning_rate": 0.05,
    }

    def __init__(
        self,
        layer_sizes: List[int] = None,
        activations: List[str] = None,
        hyper_params: Dict = None,
    ):
        """
        Initializes a neural network with the given layer sizes and activations.

        Args:
            layer_sizes: A list of integers representing the size of each layer.
            Note that the first value in the list corresponds to the input or
            the zeroth layer.
                Defaults to [12288, 7, 1].
            activations: A list of strings representing the activation function
            for each layer.
                Defaults to ['relu', 'sigmoid'].
            hyper_params: A dictionary containing hyperparameters for the neural
            network.
                Defaults to DEFAULT_HYPERPARAMS.
        """
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

    def _initialize_parameters(self, seed: int = 2023):
        """
        Initializes the parameters for each layer of the neural network.

        Args:
            seed: An integer value for the random seed. Defaults to 2023.
        """
        for layer in range(1, self.n_layers + 1):
            np.random.seed(seed)
            curr_layer = self.layer_sizes[layer]
            prev_layer = self.layer_sizes[layer - 1]
            self.parameters[f"W{layer}"] = (
                np.random.randn(curr_layer, prev_layer) * 0.01
            )
            self.parameters[f"b{layer}"] = np.zeros((curr_layer, 1))

    def _propagate_forward(self, X: np.ndarray):
        """
        Propagates input data forward through the neural network.

        Args:
            X: A numpy array of shape (n[h], n[x]) representing the
            input data.
        """
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
    def _compute_cost(AL: np.ndarray, Y: np.ndarray):
        """
        Computes the cost for the neural network. Currently uses cross-entropy.

        Args:
            AL: A numpy array of shape (1, n_samples) representing the final
            output of the neural network.
            Y: A numpy array of shape (1, n_samples) representing the expected
            output of the neural network.

        Returns:
            The cost as a scalar value.
        """
        m = Y.shape[1]
        cost = -(np.dot(Y, np.log(AL.T)) + np.dot(1 - Y, np.log(1 - AL.T))) / m

        return np.squeeze(cost)

    def _propagate_backward(self, Y: np.ndarray):
        """
        Calculates gradients of the cost with respect to weights and biases for
        each layer using backpropagation.

        Args:
            Y: The ground truth labels of shape (1, m).
        """
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
        """Update the parameters of the network using gradient descent."""
        learning_rate = self.hyper_params["learning_rate"]
        for layer in range(1, self.n_layers):
            self.parameters[f"W{layer}"] -= (
                learning_rate * self.grads[f"dW{layer}"]
            )
            self.parameters[f"b{layer}"] -= (
                learning_rate * self.grads[f"db{layer}"]
            )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        num_iterations: int = 500,
        verbose: bool = False,
    ):
        """
        Train the neural network on the input data.

        Args
            X: Input data with shape (n_features, n_samples).
            Y: True labels of the input data with shape (1, n_samples).
            num_iterations: Number of iterations to train the network,
            Defaults to 500.
            verbose: Whether to print the cost every 10 iterations,
            Defaults to False.

        """
        self._initialize_parameters()
        for iteration in range(num_iterations):
            self._propagate_forward(X)
            self._propagate_backward(Y)
            self._update_params()
            cost = self._compute_cost(self.cache[f"A{self.n_layers}"], Y)
            if verbose:
                if num_iterations % 10 == 0:
                    print(f"Cost: {cost}")

    def predict(self, X: np.ndarray):
        """
        Predict binary labels for new input data using the trained network.

        Args
        X: Input data with shape (n_features, n_samples).
        """
        self._propagate_forward(X)
        AL = self.cache[f"A{self.n_layers}"]
        return (AL > 0.5).astype(int)
