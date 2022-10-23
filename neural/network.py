# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers.logsuite import sapilog as sl
from neural import layer
import numpy as np
from random import randint
import inspect

# ===== < BODY > =====
class NNetwork:
    """
    Converts Layer objects into pure NumPy matrices for operations.
    """
    def __init__(self):
        self._raw_layers = []
        self._weights = [] # List of NumPy ndarrays representing weight matrices
        sl.log(3, f"NNetwork object created", inspect.stack())
        return

    # +----------------+
    # |    Mutators    |
    # +----------------+
    def __iadd__(self, l: layer.Layer):
        """
        Overload immediate-add (ie. addition with assignment) to allow for appending `Layer` objects.
        """
        self._raw_layers.append(l)
        sl.log(2, f"{l.__class__.__name__} layer added to NNetwork", inspect.stack())
        if len(self) >= 2:
            self._gen_weights(self._raw_layers[-2], self._raw_layers[-1])
            sl.log(2, f"{self._weights[-1].shape} weight matrix added to NNetwork", inspect.stack())
        return self

    def _gen_weights(self, from_layer: layer.Layer, to_layer: layer.Layer):
        """
        Generate a weight matrix between two layers.
        """
        w = np.random.rand(len(from_layer), len(to_layer))
        sl.log(4, f"Created randomly initialized weight matrix with shape {w.shape}", inspect.stack())
        self._weights.append(w)
        return

    def train(self, train_data: list[np.array], label_data: list[np.array], c_func: callable, q_c_func: callable, batch_size: int, n_epochs: int, learning_rate=0.1, report_freq=1):
        """
        Train the network using stochastic gradient descent.

        Parameters
        ----------
        `train_data` : list[np.array]
            List of NumPy ndarrays representing datapoints in the training pool
        `label_data` : list[np.array]
            List of NumPy ndarrays representing labels for training data
        `c_func` : callable
            Cost / loss function
        `q_c_func` : callable
            Derivative of `c_func`
        `batch_size` : int
            Number of samples per batch
        `n_epochs` : int
            Number of batches to run
        `learning_rate` : float
            Learning rate to be applied to the weight gradients
        `report_freq` : int
            How many epochs before a report is generated
        """
        # Step 0: Set up local variables and log
        avg_loss = 0
        sl.log(3, f"Beginning training with {len(train_data)} samples over {n_epochs} epochs, using batch size {batch_size} and a learning rate of {learning_rate}", inspect.stack())

        # Step 1: Main training loop
        for e in range(n_epochs):

            if e > 0 and e % report_freq == 0:
                sl.log(3, f"Epoch {e} -> avg loss={avg_loss/e}")
            batch_w_grad, batch_b_grad = [np.zeros_like(w) for w in self._weights], [np.zeros_like(l.b) for l in self._raw_layers[1:]]

            for b in range(batch_size):
                # Step 1a: Get sample
                sample_index = randint(0, len(train_data)-1)
                x, y = train_data[sample_index], label_data[sample_index]

                # Step 1b: Feed forward, and retain activations and weighted inputs
                self.feedforward(x)
                activations, zs = [l.a for l in self._raw_layers], [l.z for l in self._raw_layers[1:]]

                # Step 1c: Get the loss of the evaluation for this sample
                loss = c_func(activations[-1], y)
                avg_loss += loss

                # Step 1d: Calculate gradient for output layer
                sample_w_grad, sample_b_grad = [], []

                local_gradient = q_c_func(activations[-1], y)
                del_w = activations[-2] @ local_gradient
                del_b = local_gradient

                sample_w_grad.append(del_w)
                sample_b_grad.append(del_b)

                # Step 1e: Propagate for rest of network
                for l, w in zip(reversed(range(1, len(self._raw_layers)-1)), reversed(range(len(self._weights)))):
                    local_gradient = self._weights[w] @ local_gradient # Propagate local gradient back on weights

                    del_w = activations[l-1] @ local_gradient.T
                    del_b = local_gradient

                    sample_w_grad.insert(0, del_w)
                    sample_b_grad.insert(0, del_b)

                # Step 1f: Apply to batch gradient
                for w, dw in zip(range(len(batch_w_grad)), sample_w_grad):
                    batch_w_grad[w] = batch_w_grad[w] + dw

                for b, db in zip(range(len(batch_b_grad)), sample_b_grad):
                    batch_b_grad[b] = batch_b_grad[b] + db

            # Step 1g: Apply to weights and biases, using learning rate and averaging over batch
            for w, dw in zip(range(len(self._weights)), batch_w_grad):
                self._weights[w] = self._weights[w] - dw * (learning_rate/batch_size)

            for l, db in zip(self._raw_layers[1:], batch_b_grad):
                l.b = l.b - db * (learning_rate/batch_size)


        # Step 2: Report accuracy
        sl.log(2, f"Training complete with an average loss per epoch: {avg_loss/n_epochs}", inspect.stack())
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
        a = self._raw_layers[0].activation(x) # Force input activation
        for l, w in zip(self._raw_layers[1:], self._weights):
            a = l.activation(a, w)
        return a

    def tell_params(self):
        """
        Logs network parameters.
        """
        sl._logBox("PARAMETERS")
        sl.log(4, f"Shape: {[len(l) for l in self._raw_layers]}", inspect.stack())
        sl.log(4, f"Biases: {[l.b for l in self._raw_layers]}", inspect.stack())
        sl.log(4, f"Weights: {self._weights}", inspect.stack())
        return

    def evaluate(self, val_data: list[np.array], label_data: list[np.array], c_func: callable, threshold: float = 0.1) -> float:
        """
        Return the accuracy of the network over a labeled validation set.
        """
        n_correct = 0
        avg_cost = 0

        for x, y in zip(val_data, label_data):
            guess = self.feedforward(x)
            cost = c_func(guess, y)
            avg_cost += cost/len(val_data)
            if cost <= threshold: n_correct += 1

        sl.log(3, f"{n_correct} out of {len(label_data)} samples were within cost threshold {threshold}, for a total accuracy of {n_correct/len(label_data)} and an average cost of {avg_cost}", inspect.stack())

        return n_correct/len(label_data)

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
