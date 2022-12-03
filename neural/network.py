# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers.logsuite import sapilog as sl
from helpers.algebra import mse, q_mse
from neural import layer
import numpy as np
from random import randint
from inspect import stack

# ===== < BODY > =====
class NNetwork:
    """

    """
    def __init__(self, layers: list["Layer"] = None):
        self._layers = layers
        self._shape = []
        return

    def finalize(self, x_shape: tuple[int]):
        """
        Initialize all trainable parameters in the network.

        Parameters
        ----------
        `x_shape` : tuple[int]
            Input shape.
        """
        # Columnize ("flatten") input dimensions
        columnized_shape = 1
        for dim in x_shape: columnized_shape *= dim
        sl.log(3, f"Columnized {x_shape} to {columnized_shape}", stack())

        # Finalize layers in cascading fashion
        self._layers[0].finalize(columnized_shape)

        for l in range(1, len(self._layers)):
            self._layers[l].finalize(len(self._layers[l-1]))

        # Build shape, log and return
        self._shape = [len(l) for l in self._layers]
        self._shape.insert(0, columnized_shape)
        sl.log(2, f"Finalized model with shape {self._shape}", stack())
        return

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """

        """
        return

    def train(self, x_set: list[np.ndarray], y_set: list[np.ndarray], batch_size: int, n_epochs: int):
        """
        Train the network on a given set with contiguous labels.

        Parameters
        ----------
        `x_set` : list[np.ndarray]
            Feature set to train on.
        `y_set` : list[np.ndarray]
            Label set to train on (contiguous to `x_set`).
        `batch_size` : int
            Number of samples to consider before updating weights.
        `n_epochs` : int
            Number of times to repeat training.
        """
        # Step 1: Select a random sample
        r = randint(0, len(x_set)-1)
        x, y = x_set[r], y_set[r]
        sl.log(4, f"Considering sample {r}: {x.tolist()} -> {y}", stack())

        # Step 2: Forward propagation, loss
        y_hat = x # Set "activation" to be input sample
        for l in self._layers:
            y_hat = l.activation(y_hat) # Overwrite it with layer's activation
        loss = mse(y_hat, y)

        # Step 3: Error and weight gradient for output layer
        e = np.multiply(self._layers[-1].qa, q_mse(y_hat, y))
        w_grad = (e @ self._layers[-2].a.T).T ## Deviating from eq.3; transposing to turn rowvec into colvec
        sl.log(4, f"Output error: {e.tolist()} | weight gradient: {w_grad.tolist()}", stack())

        # Step 4: Backprop error and recalculate weight gradient (not including final layer)
        for l in reversed(range(1, len(self._layers)-1)):
            e = np.multiply(self._layers[l].qa, self._layers[l+1].error_prop(e))
            w_grad = (e @ self._layers[l-1].a.T).T ## Deviating from eq.3; transposing to turn rowvec into colvec
            sl.log(4, f"Layer {l+1} error: {e.tolist()} | weight gradient: {w_grad.tolist()}", stack())

        # Step 5: Backprop to the final (first) layer
        e = np.multiply(self._layers[0].qa, self._layers[1].error_prop(e))
        w_grad = (e @ x.T).T ## Deviating from eq.3; transposing to turn rowvec into colvec
        sl.log(4, f"Layer 1 error: {e.tolist()} | weight gradient: {w_grad.tolist()}", stack())

        sl.log(2, f"Training complete")
        return
# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
