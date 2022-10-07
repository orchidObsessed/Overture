# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers import sapilog as sl
import numpy as np

# ===== < BODY > =====
class Layer:
    """
    Base layer class.
    """
    def __init__(self, size: int, a_func: callable, q_a_func: callable):
        self._size = size
        self.b = np.zeros((size, 1)) # Sad that this needs to be mat instead of vec
        self._a_func = a_func
        self._q_a_func = q_a_func
        self.z, self.a = None, None
        sl.log(3, f"{self.__class__.__name__} created")
        return

    def activation(self, x: np.array, w: np.array) -> np.array:
        """
        Return the activation of this layer as an nx1 Numpy ndarray.

        Parameters
        ----------
        `x` : np.array
            Activations in previous layer as a NumPy ndarray of floats.
        `w` : np.array
            Weights between previous layer and here as a NumPy ndarray of floats.
        """
        self.z = w.T @ x + self.b
        self.a = self._a_func(self.z)
        return self.a

    def q_activation(self):
        """
        Return the derivative of the activation function with most recent weighted input.
        """
        if not self.z:
            sl.log(0, "Called before z value calculated")
            raise sl.SapiException()
        return self._q_a_func(self.z)

    def __len__(self):
        return self._size

class Flatten(Layer):
    """
    Flattening layer; takes an input of a given dimension and reshapes it to a column vector
    """
    def __init__(self, size: int, dim: tuple[int]):
        self._indim = dim
        self.b = None
        self._outdim = (size, 1)
        self._size = size
        self.a = None

    def activation(self, x: np.array) -> np.array:
        """
        Reshapes (if necessary) and returns the passed array (or array-like) object.
        """
        try:
            self.a = np.array(x).reshape(self._outdim)
            return self.a
        except ValueError as e:
            sl.log(0, f"Cannot reshape input {x} of dimension {x.shape} to {self._outdim}")
            raise sl.sapiDumpOnExit()

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
