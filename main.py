# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers.logsuite import sapilog as sl
from helpers import datagen
from helpers import algebra as alg
from neural import layer as l
from neural import network as nn
import numpy as np
from time import sleep

sl.MAX_V_PRINT = 4
sl.MAX_V_WRITE = 0 # Don't want to be writing to log for now
# ===== < BODY > =====

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    # TEST - Set all warnings to raise as errors instead
    # np.seterr(all="raise")

    # Read and format test data
    fold1, fold2, fold3 = datagen.collectData("d3&c25&s200")
    train_data, label_data = list(fold1.keys()), list(fold1.values())
    train_data = [np.array(x) for x in train_data] # .reshape(2, 1) on the array
    label_data = [np.array(x) for x in label_data]

    # Format validation data
    validata, valilabel = list(fold2.keys()), list(fold2.values())
    validata = [np.array(x).reshape(2, 1) for x in validata]
    valilabel = [np.array(x) for x in valilabel]

    # Create network
    test_network = nn.NNetwork()
    test_network += l.Flatten(2, (1, 2)) # Input layer
    test_network += l.Layer(2, alg.identity, alg.q_identity) # Hidden 1 - Identity
    # test_network += l.Layer(3, alg.tanh, alg.q_tanh) # Hidden 2 - Tanh
    test_network += l.Layer(1, alg.sigmoid, alg.q_sigmoid) # Output - Sigmoid

    # print(test_network.feedforward([-2, 1]))

    # Train test network
    test_network.train(train_data, label_data, alg.mse, alg.q_mse, batch_size=1, n_epochs=25, report_freq=1)

    # Evaluate network
    # test_network.evaluate(validata, valilabel, alg.mse)
    test_network.tell_params()
