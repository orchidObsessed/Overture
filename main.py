# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers.logsuite import sapilog as sl
from helpers import datagen
from helpers import algebra as alg
from neural import layer as l
from neural import network as nn
import numpy as np

sl.MAX_V_PRINT = 4 # Print everything
sl.MAX_V_WRITE = 0 # Don't want to be writing to log for now
# ===== < BODY > =====

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    # Read and format test data
    fold1, fold2, fold3 = datagen.collectData("d3&c25&s200")
    train_data, label_data = list(fold1.keys()), list(fold1.values())
    train_data = [np.array(x) for x in train_data] # .reshape(2, 1) on the array
    label_data = [np.array(x) for x in label_data]

    # Format validation data
    validata, valilabel = list(fold2.keys()), list(fold2.values())
    validata = [np.array(x).reshape(2, 1) for x in validata]
    valilabel = [np.expand_dims(np.expand_dims(x, -1), -1) for x in valilabel]

    # Create network
    test_network = nn.NNetwork([l.Dense(3),
                                l.Dense(2),
                                l.Dense(1)])
    test_network.finalize((2,))

    test_network.train(x_set=validata[0:3], y_set=valilabel[0:3], batch_size=3, n_epochs=1)
