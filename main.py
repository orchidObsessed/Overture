# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from neural import layer as l
from neural import network as nn
import numpy as np
# ===== < BODY > =====

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    test_network = nn.NNetwork()
    test_network += l.Layer(2, lambda x: x)
    test_network += l.Layer(3, lambda x: x)
    test_network += l.Layer(1, lambda x: x)

    test_x = np.array([[2], [2]]) # Sample input

    output = test_network.feedforward(test_x)
    print(f"Input {test_x}\nyields output {output}")
