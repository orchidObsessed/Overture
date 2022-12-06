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
        """
        Returns the number of nodes in this layer.
        """
        return self._size

class Conv:
    """

    """
    id = 1

    def __init__(self, kernel_shape: tuple[int] = (3, 3), n_filters: int = 1, stride: int = 1):
        # Real attributes
        self._id = Conv.id
        self._kernel_shape = list(kernel_shape) # Kernel shape; depth will be populated in finalize()
        self._filters = [] # List of filter matrices
        self._n_filters = n_filters
        self._stride = stride # Filter stride to use in _convo_slices
        self._expected_in = None # Expected input shape (row, col, depth)
        self._expected_out = None # Expected output shape (row, col, depth)

        # Cache attributes
        self._last_input = None # Most recent input

        Conv.id += 1
        return

    def finalize(self, n_before: tuple[int]) -> None:
        """
        Lock in dimensional attributes and initialize trainable parameters.

        Parameters
        ----------
        `n_before` : tuple[int]
            Output shape of previous layer.
        """
        # Build input shape
        if len(n_before) == 3:
            self._expected_in = n_before
        elif len(n_before) == 2: # If not given depth channel
            self._expected_in = (n_before[0], n_before[1], 1) # Assume depth 1
        else:
            sl.log(0, f"[Conv-{self._id}] Incorrect dimensions given for finalization: {n_before}", stack())
            raise Exception

        # Build output shape
        self._expected_out = []
        self._expected_out.append(((self._expected_in[0] - self._kernel_shape[0]) // self._stride)+1)
        self._expected_out.append(((self._expected_in[1] - self._kernel_shape[1]) // self._stride)+1)
        self._expected_out.append(self._n_filters)
        self._expected_out = tuple(self._expected_out)

        # Initialize filters
        self._kernel_shape.append(self._expected_in[-1]) # Expand filter to full depth of input
        self._kernel_shape = tuple(self._kernel_shape)
        for _ in range(self._n_filters):
            self._filters.append(np.random.rand(*self._kernel_shape, self._expected_in[-1])) # Unpack kernel shape on-the-fly
        sl.log(4, f"[Conv-{self._id}] Built {self._n_filters} filters with shape {self._kernel_shape}", stack())

        sl.log(4, f"[Conv-{self._id}] Expected dims: {self._expected_in} -> {self._expected_out}", stack())
        return

    def _convo_slices(self, prev_activation: np.ndarray) -> list[np.ndarray]:
        """

        """
        chunks = []
        for row in range(0, self._expected_in[0], self._stride):
            for col in range(0, self._expected_in[1], self._stride):
                sl.log(4, f"[MaxPool-{self._id}] slicing on [0:{self._expected_in[-1]},{row}:{row+self._kernel_shape[0]},{col}:{col+self._kernel_shape[1]}]", stack())
                chunks.append(prev_activation[0:self._expected_in[-1],row:row+self._kernel_shape[0], col:col+self._kernel_shape[1]])
        return chunks

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

    def __init__(self, kernel_shape: int, stride: int = None):
        # Real attributes
        self._id = MaxPool.id
        self._kernel_shape = (kernel_shape, kernel_shape) # Kernel dimensions (r, c)
        if stride: self._stride = stride # Stride to move filter by between each slice
        else: self._stride = kernel_shape # If no stride given, assume no-overlap stride length
        self._n_slices = None # Number of slices we can expect
        self._in_shape = None # Expected shape of input
        self._out_shape = None # Expected shape of output

        # Cache attributes
        self.a = None # Most recent output
        self._last_input = None # Most recent input

        sl.log(4, f"[MaxPool-{self._id}] Kernel dimensions: {self._kernel_shape} || Stride: {self._stride}", stack())
        MaxPool.id += 1
        return

    def finalize(self, in_shape: tuple[int]) -> None:
        """

        """
        self._in_shape = in_shape
        if in_shape[0] % self._stride != 0 or in_shape[1] % self._stride != 0:
            sl.log(1, f"[MaxPool-{self._id}] Please double-check 2D dimensions: input dimension {in_shape}, kernel dimension {self._kernel_shape}, stride {self._stride}", stack())
        self._out_shape = (in_shape[0] // self._stride, in_shape[1] // self._stride)
        return

    def _convo_slices(self, prev_activation: np.ndarray) -> list[np.ndarray]:
        """
        Builds and returns an array of subsections of the input to act on.

        Notes
        -----
        This could be rewritten to be more efficient via `yield`, but that would
        require rewriting other functions to either 1. convert the generator to
        a true list, or 2. make use of the generator in a loop.
        """
        slices = []
        for row in range(0, self._in_shape[0], self._stride):
            for col in range(0, self._in_shape[1], self._stride):
                sl.log(4, f"[MaxPool-{self._id}] slicing on [{row}:{row+2},{col}:{col+2}]", stack())
                slices.append(prev_activation[row:row+self._kernel_shape[0], col:col+self._kernel_shape[1]])
        return slices

    def activation(self, prev_activation: np.ndarray) -> np.ndarray:
        """
        Returns maximum value in each subslice of `prev_activation`, where subslices are determined by the kernel shape and stride.
        """
        self._last_input = prev_activation
        slices = self._convo_slices(prev_activation)
        output = []

        for slice in slices:
            output.append(np.amax(slice))
        self.a = np.reshape(output, self._out_shape)
        sl.log(4, f"[MaxPool-{self._id}] Activation: {self.a.tolist()}", stack())
        return self.a

    def backprop(self, front_error_gradient: np.ndarray) -> np.ndarray:
        """
        Propagate `front_error_gradient` backwards through this layer.

        This is done by returning a 0-array with shape `in_shape`, and placing each value in `front_error_gradient` into its corresponding index.

        Notes
        -----
        This is a very sloppy implementation that could probably be dramatically improved.
        """
        # Re-generate indices of max values from last activation
        e = []
        slices = self._convo_slices(self._last_input) # most recent set of subslices
        for slice, a, i in zip(slices, self.a.flatten().tolist(), range(self._out_shape[0]*self._out_shape[1])):
            # Find the index of max value
            index = np.where(slice == a)
            index = (index[0][0], index[1][0])

            # Set that index equal to corresponding error gradient, all others 0
            sub_e = np.zeros_like(slice)
            sub_e[index[0], index[1]] = front_error_gradient.flatten()[i]
            e.append(sub_e)

        # Final dimensionality & typing
        e = np.reshape(e, self._in_shape)
        sl.log(4, f"[MaxPool-{self._id}] Error: {e.tolist()}", stack())
        return e

    def adjust(self, *args, **kwargs):
        """
        This function does nothing, since `MaxPool` layers are non-trainable.
        """
        sl.log(4, f"[MaxPool-{self._id}] Doing nothing", stack())
        return

class Flatten:
    """
    Non-trainable layer; covnverts true-2D input into column matrix form.
    """
    id = 1

    def __init__(self):
        # Real attributes
        self._in_shape = None # Expected input shape
        self._out_shape = None # Expected output shape
        self._id = Flatten.id # Unique ID

        # Cache attribtes
        self.a = None # Last output
        self._prev_input = None # Last input

        Flatten.id += 1
        return

    def finalize(self, in_shape: tuple[int]) -> None:
        """
        Finalize this layer.

        This function should only be called by NNetwork, not by developers.
        """
        # Dimensionality attribute initialization
        self._in_shape = in_shape
        self._out_shape = 1
        for dim in in_shape: self._out_shape *= dim
        self._out_shape = (self._out_shape, 1)

        sl.log(4, f"[Flatten-{self._id}] Input: {self._in_shape} || Output: {self._out_shape}", stack())
        return

    def activation(self, prev_activation: np.ndarray) -> np.ndarray:
        """
        Converts 2D input to 1D output, while spatial order.
        """
        self.a = np.expand_dims(prev_activation.flatten(), -1)
        sl.log(4, f"[Flatten-{self._id}] a={self.a.tolist()}", stack())
        return self.a

    def backprop(self, front_error_gradient: np.ndarray) -> np.ndarray:
        """
        Just reshapes error gradient from a column vector to correct 2D form and passes it back.
        """
        e = np.reshape(front_error_gradient, self._in_shape)
        sl.log(4, f"[Flatten-{self._id}] e={e.tolist()}", stack())
        return e

    def adjust(self, *args, **kwargs):
        """
        Empty function.
        """
        sl.log(4, f"[Flatten-{self._id}] doing nothing", stack())
        return

    def __len__(self):
        """
        Returns the column-length of this layer's output.
        """
        return self._out_shape[0]
# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
