# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers import sapilog as sl
import numpy as np
# ===== < BODY > =====
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

def sigmoid(x):
    """

    """
    return 1 / (1 + np.exp(-x))

def q_sigmoid(x):
    """

    """
    return sigmoid(x) * (1 - sigmoid(x))

def mse(x, y):
    """

    """
    return ((x - y) ** 2).mean(axis=0)

def q_mse(x, y):
    """

    """
    return 2*(x-y)

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
