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
        self._prev_activation = None # Most recently recieved input
        self._batch_weight_gradient = None # Batch weight update gradient
        self._batch_bias_gradient = None # Batch bias update gradient

        Dense.id += 1
        return

    def finalize(self, n_before: int) -> None:
        """
        Finalize this layer by initializing weights, biases, and callables.

        This function should only be called by NNetwork, not by developers.
        """
        # Weights & biases
        self._biases, self._batch_bias_gradient = np.random.rand(self._size, 1), np.zeros((self._size, 1))
        self._weights, self._batch_weight_gradient = np.random.rand(n_before, self._size), np.zeros((n_before, self._size))
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
        self._prev_activation = prev_activation
        self.a = self._a_func((self._weights.T @ prev_activation) + self._biases)
        self.qa = self._q_a_func((self._weights.T @ prev_activation) + self._biases) # Store, since we'll likely need this (and it's inexpensive)
        sl.log(4, f"[Dense-{self._id}] a = {self.a.tolist()} | qa = {self.qa.tolist()}", stack())
        return self.a

    def backprop(self, front_error_gradient: np.ndarray) -> np.ndarray:
        """
        Calculate the hadamard term for backward error propagation, FROM this layer.

        Parameters
        ----------
        `my_error` : np.ndarray
            The backpropagated error from the next layer.

        Returns
        -------
        np.ndarry
            The matrix product of this layer's weights and this layer's error.
        """
        e = np.multiply(self.qa, front_error_gradient)
        w_grad = (e @ self._prev_activation.T).T
        self._batch_weight_gradient = self._batch_bias_gradient + w_grad
        self._batch_bias_gradient = self._batch_bias_gradient + e
        sl.log(4, f"[Dense-{self._id}] Error: {e.tolist()} || Weight gradient: {w_grad.tolist()}", stack())
        return self._weights @ front_error_gradient

    def adjust(self, learning_rate: float, batch_size: int) -> None:
        """
        Update weights and biases of this layer.

        Parameters
        ----------
        `learning_rate` : float
            Learning rate to be applied to the gradients.
        `batch_size` : int
            Batch size to use to average changes.
        """
        # Apply learning rate & batch size averaging to delta gradients
        self._batch_bias_gradient = self._batch_bias_gradient * (learning_rate/batch_size)
        self._batch_weight_gradient = self._batch_weight_gradient * (learning_rate/batch_size)
        sl.log(4, f"[Dense-{self._id}] Weight gradient = {self._batch_weight_gradient.tolist()} | Bias gradient = {self._batch_bias_gradient.tolist()}", stack())
        # Adjust weights & biases
        self._weights = self._weights - self._batch_weight_gradient
        self._biases = self._biases - self._batch_bias_gradient
        sl.log(4, f"[Dense-{self._id}] New weights = {self._weights.tolist()} | New biases = {self._biases.tolist()}", stack())
        # Reset gradients
        self._batch_bias_gradient = np.zeros_like(self._batch_bias_gradient)
        self._batch_weight_gradient = np.zeros_like(self._batch_weight_gradient)
        return

    def __len__(self):
        return self._size

class Conv:
    """

    """
    id = 1

    def __init__(self):
        # Real attributes
        self._id = Conv.id
        self._kernel_shape = None # Shape of kernel to use for activation
        self._stride = None # How much to scoot kernel by
        self._a_func = None # Activation function
        self._q_a_func = None # Derivative of activation function

        # Cache attributes
        self.a = None
        Conv.id += 1
        return

    def finalize(self, n_before: tuple[int]) -> None:
        """

        """
        return

    def activation(self):
        """

        """
        return

    def error_prop(self):
        """

        """
        return

    def adjust(self):
        """

        """
        return

class MaxPool:
    """

    """
    id = 1

    def __init__(self):
        return

    def activation(self):
        """

        """
        return

    def error_prop(self):
        """

        """
        return

    def adjust(self):
        """

        """
        return

class Flatten:
    """

    """
    id = 1

    def __init__(self):
        return

    def activation(self):
        """

        """
        return

    def error_prop(self):
        """

        """
        return

    def adjust(self):
        """

        """
        return
# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
