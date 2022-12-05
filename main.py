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
    train_data, train_label = list(fold1.keys()), list(fold1.values())
    train_data = [np.array(x).reshape(2, 1) for x in train_data] # .reshape(2, 1) on the array
    train_label = [np.expand_dims(np.expand_dims(x, -1), -1) for x in train_label]

    # Format validation data
    test_data, test_label = list(fold2.keys()), list(fold2.values())
    test_data = [np.array(x).reshape(2, 1) for x in test_data]
    test_label = [np.expand_dims(np.expand_dims(x, -1), -1) for x in test_label]

    # Create network
    # test_network = nn.NNetwork([l.Dense(2),
    #                             l.Dense(1, a_func=alg.sigmoid, q_a_func=alg.q_sigmoid)])
    # test_network.finalize((2,))
    test_mp = l.MaxPool(2, 2)
    test_mp.finalize((4, 4))
    sample_im = np.reshape([1, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4], (4, 4))
    sample_e = np.reshape([1, 2, 3, 4], (2, 2))
    # slices = test_mp._convo_slices(sample_im)
    test_mp.activation(sample_im)
    test_mp.backprop(sample_e)
    test_mp.adjust(learning_rate=500, batch_size=10)


    # Train network
    # test_network.train(x_set=train_data, y_set=train_label, batch_size=128, n_epochs=10, learning_rate=0.05)

    # Evaluate network
    # higuess = test_network.predict(test_data[0])
    # loguess = test_network.predict(test_data[1])
    # print(f"Got {higuess}, expected {test_label[0]}")
    # print(f"Got {loguess}, expected {test_label[1]}")
    # test_network.evaluate(test_data, test_label)
