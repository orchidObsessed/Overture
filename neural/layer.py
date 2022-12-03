# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers.logsuite import sapilog as sl
import numpy as np
from inspect import stack

# ===== < BODY > =====
class Dense:
    """
    Standard densely-connected layer.
    """
    id = 1

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
        self._weights = np.random.rand(n_before, self._size)
        sl.log(4, f"[Dense-{self._id}] w.shape = {self._weights.shape} | b.shape = {self._biases.shape}", stack())

        # If no functions given, use identity
        if not self._a_func:
            self._a_func = lambda x: x
            sl.log(4, f"[Dense-{self._id}] defaulting to identity activation function", stack())
        if not self._q_a_func:
            self._q_a_func = lambda x: np.ones_like(x)
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
        sl.log(4, f"[Dense-{self._id}] a = {self.a.tolist()} | qa = {self.qa.tolist()}", stack())
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
        sl.log(4, f"[Dense-{self._id}] error backprop term = {e.tolist()}", stack())
        return e

    def adjust(self, w_grad: np.ndarray, b_grad: np.ndarray) -> None:
        """
        Update weights and biases of this layer.

        Parameters
        ----------
        `w_grad` : np.ndarray
            Weight gradient to update with. Learning rate should be pre-applied!
        `b_grad` : np.ndarray
            Bias gradient to update with.
        """
        sl.log(4, f"[Dense-{self._id}] w_grad = {w_grad.tolist()} | b_grad = {b_grad.tolist()}", stack())
        self._weights = self._weights - w_grad
        self._biases = self._biases - b_grad
        return

    def __len__(self):
        return self._size

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
