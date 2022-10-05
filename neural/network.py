# ===== < INFO > =====
# +----------------+
# | |
# +----------------+
# ===== < IMPORTS & CONSTANTS > =====
from helpers import sapilog as sl
from neural import layer
import numpy as np
# ===== < BODY > =====
class NNetwork:
    """
    Converts Layer objects into pure NumPy matrices for operations.
    """
    def __init__(self):
        self._raw_layers = []
        self._weights = [] # List of NumPy ndarrays representing weight matrices

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

        """
        a = x # Force input activation
        for l, w in zip(self._raw_layers[1:], self._weights):
            a = l.activation(a, w)
        return a

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
