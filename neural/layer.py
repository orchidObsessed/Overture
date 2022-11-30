# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers.logsuite import sapilog as sl
import numpy as np
import inspect

# ===== < BODY > =====
class Layer:
    """
    Base layer class. This is the same as a Dense layer (may be separated in future).
    """
    def __init__(self, size: int, a_func: callable, q_a_func: callable):
        self._size = size
        self.b = np.zeros((size, 1)) # Sad that this needs to be mat instead of vec
        self._a_func = a_func
        self._q_a_func = q_a_func
        self.z, self.a = None, None
        sl.log(3, f"{self.__class__.__name__} created", inspect.stack())
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
        sl.log(4, f"Finding activation of {self.__class__.__name__} with x = {x} and w = {w}", inspect.stack())
        self.z = w.T @ x + self.b
        self.a = self._a_func(self.z)
        return self.a

    def q_activation(self):
        """
        Return the derivative of the activation function with most recent weighted input.
        """
        if not self.z:
            sl.log(0, "Called before z value calculated", inspect.stack())
            raise sl.SapiException()
        return self._q_a_func(self.z)

    def __len__(self):
        return self._size

class Flatten(Layer):
    """
    Flattening layer; takes an input of a given dimension and reshapes it to a column vector.
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
            sl.log(0, f"Cannot reshape input {x} of dimension {x.shape} to {self._outdim}", inspect.stack())
            raise sl.sapiDumpOnExit()

class Dense:
    """

    """
    id = 0

    def __init__(self, n_nodes: int, a_func: callable = None, q_a_func: callable = None) -> None:
        # Real attributes
        self._size = n_nodes # Size of layer
        self._id = Dense.id # Unique ID for log purposes
        self._biases = None # Biases for this layer
        self._weights = None # Weights (incoming) for this layer
        self._a_func = a_func # Activation function
        self._q_a_func = q_a_func # Derivative of activation function

        # Cache attributes
        self.a = None # Activation
        self.qa = None # Derivative of activation

        Dense.id += 1
        return

    def finalize(self, n_before: int) -> None:
        """
        Finalize this layer by initializing weights, biases, and callables.

        This function should only be called by NNetwork, not by developers.
        """
        # Weights & biases
        self._biases = np.random.rand(self._size, 1)
        self._weights = np.random.rand(shape=(n_before, self._size))

        # If no functions given, use identity
        if not self._a_func:
            self._a_func = lambda x: x
        if not self._q_a_func:
            self._a_func = lambda x: np.ones_like(x)
        return
    
    def activation(self, prev_activation: np.ndarray) -> np.ndarray:
        """
        Calculate the activation function for this layer.

        Parameters
        ----------
        `prev_activation` : numpy.ndarray
            The activation of the previous layer, as a column vector.

        Returns
        -------
        numpy.ndarray
            The activation for this layer as a column vector.
        """
        self.a = self._a_func((self._weights.T @ prev_activation) + self._biases)
        self.qa = self._q_a_func((self._weights.T @ prev_activation) + self._biases) # Store, since we'll likely need this (and it's inexpensive)
        sl.log(4, f"[Dense-{self._id}] a = {self.a.tolist()} | qa = {self.qa.tolist()}")
        return self.a

    def error_prop(self, my_error: np.ndarray) -> np.ndarray:
        """
        Calculate the hadamard term for backward error propagation, FROM this layer.

        Parameters
        ----------
        my_error : np.ndarray
            The error gradient at this layer.

        Returns
        -------
        np.ndarry
            The matrix product of this layer's weights and this layer's error.
        """
        e = self._weights @ my_error
        sl.log(4, f"[Dense-{self._id}] error backprop term = {e}")
        return e

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
