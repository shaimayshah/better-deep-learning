import numpy as np
import pytest

from src.models.neural_network import NeuralNetwork


@pytest.fixture
def nn():
    return NeuralNetwork(
        layer_sizes=[2, 3, 1],
        activations=["relu", "sigmoid"],
        hyper_params={"learning_rate": 0.1},
    )


def test_initialize_parameters(nn):
    nn._initialize_parameters(seed=2021)
    assert np.isclose(np.sum(nn.parameters["W1"]), 0.0079001928659031)
    assert np.isclose(np.sum(nn.parameters["b1"]), 0.0)
    assert np.isclose(np.sum(nn.parameters["W2"]), 0.017461685545951865)
    assert np.isclose(np.sum(nn.parameters["b2"]), 0.0)


def test_propagate_forward(nn):
    nn._initialize_parameters(seed=2021)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).T
    nn._propagate_forward(X)
    assert nn.cache["Z1"].shape == (3, 3)
    assert nn.cache["A1"].shape == (3, 3)
    assert nn.cache["Z2"].shape == (1, 3)
    assert nn.cache["A2"].shape == (1, 3)


def test_compute_cost(nn):
    AL = np.array([[0.9, 0.1, 0.8]])
    Y = np.array([[1, 0, 1]])
    cost = nn._compute_cost(AL, Y)
    assert np.isclose(cost, 0.14462153)


def test_propagate_backward(nn):
    nn._initialize_parameters(seed=2021)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).T
    Y = np.array([[1, 0, 1]])
    nn._propagate_forward(X)
    nn._propagate_backward(Y)
    assert nn.grads["dW1"].shape == (3, 2)
    assert nn.grads["db1"].shape == (3, 1)
    assert nn.grads["dW2"].shape == (1, 3)
    assert nn.grads["db2"].shape == (1, 1)
