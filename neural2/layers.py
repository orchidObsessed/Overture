# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
import numpy as np
from helpers.logsuite import sapilog as sl
# ===== < BODY > =====
class Layer:
    """
    Non-instantiable Layer class.

    Only used for overloading operators to prevent bloat in other classes, and to allow for is-a relationships with this object.
    """
    pass

class Dense(Layer):
    """
    Standard, fully-connected, feed-forward layer.
    """
    pass

class Input(Layer):
    """
    Non-trainable layer. Only used to explicitly specify shape of input.
    """
    pass

class Flatten(Layer):
    """
    Non-trainable layer.

    Reduces dimensionality of activation to an nx1 column vector.
    """
    pass

class Convolve(Layer):
    """
    Convolutional layer.
    """
    pass

# ===== < MAIN > =====
if __name__ == "__main__":
    print("ok")
