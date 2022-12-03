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

    def predict(self, x: np.ndarray) -> np.ndarray:
        """

        """
        y_hat = x # Set "activation" to be input sample
        for l in self._layers:
            y_hat = l.activation(y_hat) # Overwrite it with layer's activation
        return y_hat

    def evaluate(self, x_set: list[np.ndarray], y_set: list[np.ndarray]) -> np.ndarray:
        """

        """
        avgloss = 0
        for x, y in zip(x_set, y_set):
            prediction = self.predict(x)
            avgloss += mse(prediction, y)
        avgloss /= len(x_set)
        sl.log(2, f"Average loss: {avgloss}", stack())
        return avgloss

    def train(self, x_set: list[np.ndarray], y_set: list[np.ndarray], batch_size: int, n_epochs: int, learning_rate: int = 0.001):
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
        for epoch in range(n_epochs):
            for n_batches in range(int(len(x_set) / batch_size)):
                batch_w_grad, batch_b_grad = [np.zeros_like(l._weights) for l in self._layers], [np.zeros_like(l._biases) for l in self._layers]
                for _ in range(batch_size):
                    # Step 1: Select a random sample
                    r = randint(0, len(x_set)-1)
                    x, y = x_set[r], y_set[r]
                    sl.log(4, f"Considering sample {r}: {x.tolist()} -> {y}", stack())

                    # Step 2: Forward propagation, loss
                    y_hat = x # Set "activation" to be input sample
                    for l in self._layers:
                        y_hat = l.activation(y_hat) # Overwrite it with layer's activation
                    loss = mse(y_hat, y)

                    # Step 3: Error and weight gradient for output layer - only do this step if L > 1
                    e = np.multiply(self._layers[-1].qa, q_mse(y_hat, y))
                    if len(self._layers) > 1:
                        w_grad = (e @ self._layers[-2].a.T).T ## Deviating from eq.3; transposing to turn rowvec into colvec
                        batch_w_grad[-1] = batch_w_grad[-1] + w_grad
                        sl.log(4, f"Output error: {e.tolist()} | weight gradient: {w_grad.tolist()}", stack())

                    # Step 4: Backprop error and recalculate weight gradient (not including final layer)
                    for l in reversed(range(1, len(self._layers)-1)):
                        e = np.multiply(self._layers[l].qa, self._layers[l+1].error_prop(e))
                        w_grad = (e @ self._layers[l-1].a.T).T ## Deviating from eq.3; transposing to turn rowvec into colvec
                        batch_w_grad[l] = batch_w_grad[l] + w_grad
                        sl.log(4, f"Layer {l+1} error: {e.tolist()} | weight gradient: {w_grad.tolist()}", stack())

                    # Step 5: Backprop to the final (first) layer
                    if len(self._layers) > 1: e = np.multiply(self._layers[0].qa, self._layers[1].error_prop(e)) # don't backprop if L == 1
                    w_grad = (e @ x.T).T ## Deviating from eq.3; transposing to turn rowvec into colvec
                    batch_w_grad[0] = batch_w_grad[0] + w_grad
                    sl.log(4, f"Layer 1 error: {e.tolist()} | weight gradient: {w_grad.tolist()}", stack())

                batch_w_grad = [g * learning_rate for g in batch_w_grad]
                sl.log(3, f"Final gradients: {[x.tolist() for x in batch_w_grad]}")
                for l, grad in zip(self._layers, batch_w_grad):
                    l.adjust(grad)

        sl.log(2, f"Training complete")
        return
# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
