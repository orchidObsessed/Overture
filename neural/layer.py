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
        self._b = np.zeros((size, 1)) # Sad that this needs to be mat instead of vec
        self._a_func = a_func
        self._q_a_func = q_a_func
        self.z, self.a = None, None
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
        sl.log(4, f"Calculating activation with input {x} and weight matrix {w}")
        self.z = w.T @ x + self._b
        self.a = self._a_func(self.z)
        sl.log(4, f"Created z={self.z} and a={self.a}")
        return self.a

    def __len__(self):
        return self._size

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
