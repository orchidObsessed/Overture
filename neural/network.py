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
    Template neural network class.
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
        sl.log(4, f"Columnized {x_shape} to {columnized_shape}", stack())

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
        Pass a sample through the network and return its output.

        Parameters
        ----------
        `x` : np.ndarray
            Input to pass to network.

        Returns
        -------
        np.ndarray
            Network's output, as the activation of the final layer.
        """
        y_hat = x # Set "activation" to be input sample
        for l in self._layers:
            y_hat = l.activation(y_hat) # Overwrite it with layer's activation
        return y_hat

    def evaluate(self, x_set: list[np.ndarray], y_set: list[np.ndarray]) -> np.ndarray:
        """
        Determine the average loss and accuracy of the network on a given dataset.

        Parameters
        ----------
        `x_set` : list[np.ndarray]
            Set of sample inputs; must be contiguous with `y_set`.
        `y_set` : list[np.ndarray]
            Set of labels.

        Returns
        -------
        np.ndarray
            Average loss of the network.
        """
        sl.log(3, f"Evaluating network on {len(x_set)} samples...", stack())
        avgloss = 0
        for x, y in zip(x_set, y_set):
            prediction = self.predict(x)
            avgloss += mse(prediction, y)
        avgloss /= len(x_set)
        sl.log(2, f"Average loss: {avgloss}", stack())
        return avgloss

    def train(self, x_set: list[np.ndarray], y_set: list[np.ndarray], batch_size: int, n_epochs: int = 1, learning_rate: float = 0.001):
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
        `learning_rate` : float, default=0.001
            Learning rate to apply to gradients before updating parameters.
        """
        for epoch in range(n_epochs):
            sl.log(3, f"Epoch {epoch+1}/{n_epochs}", stack())
            avgloss = 0
            for n_batches in range(int(len(x_set) / batch_size)):
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
                    avgloss += loss

                    # Step 3: Backprop 2.0!
                    error_gradient = np.multiply(self._layers[-1].qa, q_mse(y_hat, y)) # output layer's error is statically generated
                    for layer in reversed(self._layers):
                        error_gradient = layer.backprop(error_gradient)

                for layer in self._layers:
                    layer.adjust(learning_rate=learning_rate, batch_size=batch_size)
                sl.log(4, f"Batch {n_batches+1} complete", stack())
            avgloss /= len(x_set)
            sl.log(3, f"Average loss for epoch {epoch+1}: {avgloss}", stack())
        sl.log(2, f"Training complete", stack())
        return

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
