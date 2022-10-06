# ===== < INFO > =====
# +----------------+
# | |
# +----------------+
# ===== < IMPORTS & CONSTANTS > =====
from helpers import sapilog as sl
from neural import layer
import numpy as np
from random import randint
# ===== < BODY > =====
class NNetwork:
    """
    Converts Layer objects into pure NumPy matrices for operations.
    """
    def __init__(self):
        self._raw_layers = []
        self._weights = [] # List of NumPy ndarrays representing weight matrices
        return

    # +----------------+
    # |    Mutators    |
    # +----------------+
    def __iadd__(self, l: layer.Layer):
        """
        Overload immediate-add (ie. addition with assignment) to allow for appending `Layer` objects.
        """
        self._raw_layers.append(l)
        if len(self) >= 2: self._gen_weights(self._raw_layers[-2], self._raw_layers[-1])
        return self

    def _gen_weights(self, from_layer: layer.Layer, to_layer: layer.Layer):
        """
        Generate a weight matrix between two layers.
        """
        w = np.random.rand(len(from_layer), len(to_layer))
        sl.log(4, f"Created randomly initialized weight matrix with shape {w.shape}")
        self._weights.append(w)
        return

    def train(self, train_data: list[np.array], label_data: list[np.array], c_func: callable, q_c_func: callable, batch_size: int, n_epochs: int, learning_rate=0.1, report_freq=1):
        """

        """
        # Step 0: Set up local variables and log
        sl.log(3, f"Beginning training with {len(train_data)} samples over {n_epochs} epochs, using batch size {batch_size} and a learning rate of {learning_rate}")

        # Step 1: Main training loop
        for e in range(n_epochs):
            if e % report_freq == 0: print(f"Epoch {e}")
            # Step 1a: Get sample
            sample_index = randint(0, len(train_data)-1)
            x, y = train_data[sample_index], label_data[sample_index]
            sl.log(4, f"Considering sample {sample_index}: {x} -> {y}")

            # Step 1b: Feed forward, and retain activations and weighted inputs
            self.feedforward(x)
            activations, zs = [l.a for l in self._raw_layers[1:]], [l.z for l in self._raw_layers[1:]]
            activations.insert(0, x)

            # Step 1c: Get the loss of the evaluation for this sample
            loss = c_func(activations[-1], y)

            # Step 1d: Calculate gradient for output layer
            sample_w_grad = []

            local_gradient = q_c_func(activations[-1], y)
            del_w = activations[-2] @ local_gradient

            sample_w_grad.append(del_w)

            # Step 1e: Propagate for rest of network
            for l, w in zip(reversed(range(1, len(self._raw_layers)-1)), reversed(range(len(self._weights)))):
                local_gradient = self._weights[w] @ local_gradient # Propagate local gradient back on weights

                del_w = activations[l-1] @ local_gradient.T
                sample_w_grad.insert(0, del_w)

            # Step 1f: Apply with learning rate to weights (should be changed to apply to megabatch)
            for w, dw in zip(range(len(self._weights)), sample_w_grad):
                self._weights[w] = self._weights[w] - dw * learning_rate


        # Step 2: Report accuracy
        n_correct = 0
        for i in range(50):
            sample_index = randint(0, len(train_data))
            x, y = train_data[sample_index], label_data[sample_index]

            guess = self.feedforward(x)
            if c_func(guess, y) < 0.1: n_correct += 1
            else: print(f"Incorrect: {x} -> {y}, evaluation: {guess}")

        sl.log(3, f"{n_correct} out of 50 samples evaluated successfully")

        return

    # +----------------+
    # |   Accessors    |
    # +----------------+
    def __len__(self):
        """
        Overload len to allow for easier representation of network.
        """
        return len(self._raw_layers)

    def feedforward(self, x: np.array):
        """
        Feed a value through the network, and return the final layer's activation as a NumPy nx1 ndarray.
        """
        a = x # Force input activation
        for l, w in zip(self._raw_layers[1:], self._weights):
            a = l.activation(a, w)
        return a

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
