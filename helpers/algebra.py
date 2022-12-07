# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
import numpy as np
import warnings, inspect
from helpers.logsuite import sapilog as sl

warnings.filterwarnings("ignore")
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
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try: return np.tanh(x)
        except Warning as w:
            sl.log(1, f"{w} with parameter x = {x} | allowing NumPy to handle silently", inspect.stack())
    return np.tanh(x)

def q_tanh(x):
    """
    Derivative of the hyperbolic tangent function.

    Applies element-wise to the passed NumPy ndarray.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try: return 1 - tanh(x)**2
        except Warning as w:
            sl.log(1, f"{w} with parameter x = {x} | allowing NumPy to handle silently", inspect.stack())
    return 1 - tanh(x)**2

def sigmoid(x):
    """
    Sigmoid function.

    Applies element-wise to the passed NumPy ndarray.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try: return 1 / (1 + np.exp(-x))
        except Warning as w:
            # sl.log(1, f"{w} with parameter x = {x} | allowing NumPy to handle silently", inspect.stack())
            pass
    return 1 / (1 + np.exp(-x))

def q_sigmoid(x):
    """
    Derivative of the sigmoid function.

    Applies element-wise to the passed NumPy ndarray.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try: return sigmoid(x) * (1 - sigmoid(x))
        except Warning as w:
            # sl.log(1, f"{w} with parameter x = {x} | allowing NumPy to handle silently", inspect.stack())
            pass
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    """
    Rectified linear unit function.

    Applies element-wise to the passed NumPy ndarray.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try: return np.maximum(0, x)
        except Warning as w:
            sl.log(1, f"{w} with parameter x = {x} | allowing NumPy to handle silently", inspect.stack())
    return np.maximum(0, x)

def q_relu(x):
    """
    Derivative of the rectified linear unit function. function.

    Applies element-wise to the passed NumPy ndarray.
    """
    x[x<=0] = 0
    x[x>1] = 1
    return x

def softmax(x):
    """
    Softmax activation function. Using numerically stable version.

    Applied element-wise to the passed NumPy ndarray.
    """
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)

def q_softmax(x):
    """
    Derivative of the softmax activation function.

    Applies element-wise to the passed NumPy ndarray.
    """
    sl.log(0, f"This function is not yet implemented!", stack())
    raise Exception

def mse(x, y):
    """
    Mean squared error loss function.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try: return ((x - y) ** 2).mean(axis=0)
        except Warning as w:
            sl.log(1, f"{w} with parameter x = {x} | allowing NumPy to handle silently", inspect.stack())
    return ((x - y) ** 2).mean(axis=0)

def q_mse(x, y):
    """
    Derivative of the mean squared error loss function.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try: return 2*(x-y)
        except Warning as w:
            sl.log(1, f"{w} with parameter x = {x} | allowing NumPy to handle silently", inspect.stack())
    return 2*(x-y)

# ===== < MAIN > =====
if __name__ == "__main__":
    x = np.matrix([[2, 2], [2, 2]])
    y = np.matrix([[2, 1], [1, 0]])
    w = np.matrix([[1, 2], [4, 5], [3, 9]])
    print(mse_l2(x, y, w))
    print(q_mse_l2(x, y, w))
