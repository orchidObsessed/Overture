# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
import numpy as np
# ===== < BODY > =====
# Activation functions
def identity(x):
    """
    Identity function.

    Applies element-wise to the passed NumPy ndarray.
    """
    return x

def q_identity(x):
    """
    Derivative of the identity function.

    Applies element-wise to the passed NumPy ndarray.
    """
    return np.ones_like(x)

def tanh(x):
    """
    Hyperbolic tangent function.

    Applies element-wise to the passed NumPy ndarray.
    """
    return np.tanh(x)

def q_tanh(x):
    """
    Derivative of the hyperbolic tangent function.

    Applies element-wise to the passed NumPy ndarray.
    """
    return 1 - tanh(x)**2

def sigmoid(x):
    """
    Sigmoid function.

    Applies element-wise to the passed NumPy ndarray.
    """
    return 1 / (1 + np.exp(-x))

def q_sigmoid(x):
    """
    Derivative of the sigmoid function.

    Applies element-wise to the passed NumPy ndarray.
    """
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    """
    Rectified linear unit function.

    Applies element-wise to the passed NumPy ndarray.
    """
    return np.maximum(0, x)

def q_relu(x):
    """
    Derivative of the rectified linear unit function. function.

    Applies element-wise to the passed NumPy ndarray.
    """
    x[x<=0] = 0
    x[x>1] = 1
    return x

def mse(x, y):
    """
    Mean squared error loss function.
    """
    return ((x - y) ** 2).mean(axis=0)

def q_mse(x, y):
    """
    Derivative of the mean squared error loss function.
    """
    return 2*(x-y)

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
