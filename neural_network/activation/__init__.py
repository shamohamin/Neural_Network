import numpy as np
# from numba import jit

# @jit(nopython=True, parallel=True)
def sigmoid(input):
    return 1. / (1 + np.exp(-input))


# @jit(nopython=True, parallel=True, fastmath=True)
def tansig(input):
    return (2. / (1 + np.exp(-2 * input))) - 1

# @jit(nopython=True, parallel=True)
def linear(input):
    return input

# @jit(nopython=True, parallel=True)
def relu(input):
    return np.where(input >= 0, input, 0)


def activation_detection(activation_func: str) -> np.ndarray:
    if activation_func == "sigmoid":
        return sigmoid
    elif activation_func == "tanh":
        return tansig
    elif activation_func == "linear":
        return linear
    elif activation_func == "relu":
        return relu
    else:
        raise Exception("Activation Function Is Not Supported!!!!!")


def drivative_of_activation(input: np.ndarray, activation_func: str) -> np.ndarray:
    if activation_func == "sigmoid":
        o = activation_detection(activation_func)(input)
        return o * (1 - o)
    elif activation_func == "tanh":
        o = activation_detection(activation_func)(input)
        return 1 - (o ** 2)
    elif activation_func == "linear":
        return np.ones(input.shape)
    elif activation_func == "relu":
        return np.where(input >= 0, 1, 0)
    else:
        raise Exception("Activation Function Is Not Supported!!!!!")

def initilizing(
    kernel_initializing: str, neurons: int, input_dim: int, include_bias: bool=True
) -> tuple:
    W, B = None, None
    if kernel_initializing == "Xavier" or kernel_initializing == "Glorot":
        if include_bias:
            return np.random.randn(neurons, input_dim) * np.sqrt(1/input_dim),\
                        np.random.randn(neurons, 1) * np.sqrt(1/input_dim)
        else:
            return np.random.randn(neurons, input_dim) * np.sqrt(1/input_dim)
    elif kernel_initializing == "he_normal":
        if include_bias:
            return np.random.randn(neurons, input_dim) * np.sqrt(2/input_dim),\
                        np.random.randn(neurons, 1) * np.sqrt(2/input_dim)
        else:
            return np.random.randn(neurons, input_dim) * np.sqrt(2/input_dim)
    else:
        raise Exception("kernel_initialization not supported")