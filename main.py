# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
from helpers.logsuite import sapilog as sl
from helpers import datagen
from helpers import algebra as alg
from neural import layer as l
from neural import network as nn
import numpy as np
from inspect import stack

# Importing Keras JUST to get the MNIST10 dataset.
from keras.datasets import mnist

sl.MAX_V_PRINT = 3 # Print everything
sl.MAX_V_WRITE = 0 # Don't want to be writing to log for now
# ===== < BODY > =====

# ===== < HELPERS > =====

# ===== < MAIN > =====
if __name__ == "__main__":
    # # Read and format datagen test data
    # fold1, fold2, fold3 = datagen.collectData("d3&c25&s200")
    # train_data, train_label = list(fold1.keys()), list(fold1.values())
    # train_data = [np.array(x).reshape(2, 1) for x in train_data] # .reshape(2, 1) on the array
    # train_label = [np.expand_dims(np.expand_dims(x, -1), -1) for x in train_label]
    #
    # # Format validation data
    # test_data, test_label = list(fold2.keys()), list(fold2.values())
    # test_data = [np.array(x).reshape(2, 1) for x in test_data]
    # test_label = [np.expand_dims(np.expand_dims(x, -1), -1) for x in test_label]

    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    # Condition training data
    train_data, train_label = list(train_data), list(train_label)
    train_data = [np.expand_dims(x, 0) for x in train_data]
    for i in range(len(train_label)):
        temp = np.zeros((10, 1))
        temp[train_label[i]] = 1
        train_label[i] = temp

    # Condition testing data
    test_data, test_label = list(test_data), list(test_label)
    test_data = [np.expand_dims(x, 0) for x in test_data]
    for i in range(len(test_label)):
        temp = np.zeros((10, 1))
        temp[test_label[i]] = 1
        test_label[i] = temp

    # Create network
    test_network = nn.NNetwork([l.Conv(kernel_shape=(2, 2), stride=2),
                                l.MaxPool(kernel_shape=2, stride=2),
                                l.Flatten(),
                                l.Dense(16, a_func=alg.sigmoid, q_a_func=alg.q_sigmoid),
                                l.Dense(10, a_func=alg.sigmoid, q_a_func=alg.q_sigmoid)])
    test_network.finalize((28, 28, 1))
    # sl.log(2, f"Predicting on mnist[0] -> {train_label[0]}: {test_network.predict(train_data[0])}", stack())

    # Train network
    # sl.log(0, str(train_data[0:10]))
    test_network.train(x_set=train_data[0:320], y_set=train_label[0:320], batch_size=64, n_epochs=1, learning_rate=0.01)

    # Evaluate network
    for i in range(5):
        guess = test_network.predict(test_data[i])
        print(f"Got {guess.tolist()}, expected {test_label[i].tolist()}")
    test_network.evaluate(test_data[0:100], test_label[0:100])
    #
    # for layer in  test_network._layers:
    #     sl.log(3, f"Layer-{layer._id} w={layer._weights.tolist()} | b={layer._biases.tolist()}")


    # Build test conv layer
    # x4, for a 4x4x4 matrix
    # 1 1 2 2
    # 1 1 2 2
    # 3 3 4 4
    # 3 3 4 4
