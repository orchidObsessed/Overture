# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers import sapilog as sl
from helpers import datagen
from helpers import algebra as alg
from neural import layer as l
from neural import network as nn
import numpy as np
from time import sleep

sl.vtalk = 4
sl.vwrite = 4
# ===== < BODY > =====

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    # Read and format test data
    fold1, fold2, fold3 = datagen.collectData("d3&c25&s200")
    train_data, label_data = list(fold1.keys()), list(fold1.values())
    train_data = [np.array(x).reshape(2, 1) for x in train_data]
    label_data = [np.array(x) for x in label_data]

    # Format validation data
    validata, valilabel = list(fold2.keys()), list(fold2.values())
    validata = [np.array(x).reshape(2, 1) for x in validata]
    valilabel = [np.array(x) for x in valilabel]

    # Create network
    test_network = nn.NNetwork()
    test_network += l.Layer(2, alg.sigmoid, alg.q_sigmoid)
    # test_network += l.Layer(3, alg.sigmoid, alg.q_sigmoid)
    test_network += l.Layer(1, alg.sigmoid, alg.q_sigmoid)

    # Train test network
    test_network.tell_params()
    test_network.train(train_data, label_data, alg.mse, alg.q_mse, 50, 1000, report_freq=25)
    test_network.tell_params()

    # Evaluate network
    test_network.evaluate(validata, valilabel, alg.mse)
