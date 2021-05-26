import numpy as np
import matplotlib.pyplot as plt
from neural_network.train import SGDNetwork, BatchNetwork
from neural_network.layer import make_layers, Layer
from neural_network.preprocess import (MIN_MAX_SCALAR, make_train_set,
                                       apply_k_fold, scaling)
from neural_network.utility_functions import fx
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.random.seed(41)
scaller = MinMaxScaler()
scaller_y = None
scaller_x = StandardScaler()


def make_train_input_data() -> tuple:
    X = np.linspace(0, 2, num=500)  # feature 1
    Y = np.linspace(0, 2, num=500)  # feature 2
    labels = fx(X, Y)
    # X = (X - np.mean(X)) / np.std(X)
    # Y = (Y - np.mean(Y)) / np.std(Y)

    # labels = (labels - np.min(labels.ravel())) / (np.max(labels.ravel()) - np.min(labels.ravel()))
    global scaller_y
    scaller_y = scaller.fit(labels.reshape(-1, 1))
    labels = scaller_y.transform(labels.reshape(-1, 1))
    labels = labels.ravel()
    # labels = (labels - np.mean(labels.ravel())) / np.std(labels.ravel())
    (training_set_samples, test_set_samples) = 370, 450

    X_train = make_train_set(
        [X[:training_set_samples], Y[:training_set_samples]])
    Y_train = labels[:training_set_samples].reshape(1, -1)

    global scaller_x
    scaller_x = scaller_x.fit(X_train.reshape(-1, 2))
    X_train = scaller_x.transform(X_train.reshape(-1, 2))
    X_train = X_train.reshape(X_train.shape[-1::-1])

    X_valid = make_train_set([X[training_set_samples:test_set_samples],
                             Y[training_set_samples:test_set_samples]])
    y_valid = labels[training_set_samples:test_set_samples].reshape(1, -1)
    X_valid = scaller_x.transform(
        X_valid.reshape(-1, 2)).reshape(X_valid.shape)

    X_test = make_train_set([X[test_set_samples:], Y[test_set_samples:]])
    Y_test = labels[test_set_samples:].reshape(1, -1)
    X_test = scaller_x.transform(
        X_test.reshape(-1, 2)).reshape(X_test.shape)
    # 2, number of test samples # 2, number of valid samples # 2, number of test samples

    # X_train, Y_train, X_valid, y_valid =  apply_k_fold(X_train, Y_train)

    return X_train, Y_train, X_valid, y_valid, X_test, Y_test, X, Y, labels


def config_layers(X_train) -> list:
    layers_config = [{
        Layer.NEURON: X_train.shape[0],
        Layer.INPUT_DIM: 0,
        Layer.LAYER_NUMBER: 0,
        Layer.IS_INPUT_LAYER: True
    }, {
        Layer.NEURON: 16,
        Layer.INPUT_DIM: 2,
        Layer.LAYER_NUMBER: 1,
        Layer.ACTIVATION: "tanh",  # "tanh",
        Layer.IS_INPUT_LAYER: False
    },
    {
        Layer.NEURON: 8,
        Layer.INPUT_DIM: 16,
        Layer.LAYER_NUMBER: 2,
        Layer.ACTIVATION: "relu",
        Layer.IS_INPUT_LAYER: False
    },
    {
        Layer.NEURON: 4,
        Layer.INPUT_DIM: 8,
        Layer.LAYER_NUMBER: 3,
        Layer.ACTIVATION: "relu",
        Layer.IS_INPUT_LAYER: False
    },
    {
        Layer.NEURON: 1,
        Layer.INPUT_DIM: 4,
        Layer.LAYER_NUMBER: 4,
        Layer.ACTIVATION: "linear",
        Layer.IS_INPUT_LAYER: False
    }]
    layers = make_layers(layers_config)

    return layers

def calling_SGD(layers, X_train, Y_train, X_valid, y_valid, lrate, epochs):
    network = SGDNetwork(layers=layers)
    
    # 0.0005  1 layer 1800 seed 41
    # 0.001 2 layer 1810 seed 41
    # 0.001 3 layer 2100 seed 41 
    history = network.fit(lrate, epochs, X_train, Y_train, X_valid, y_valid)
    loss = network.evalute(X_test, Y_test)
    
    return loss, history, network


def calling_BGD(layers, X_train, Y_train, X_valid, y_valid, lrate, epochs):
    network = BatchNetwork(layers=layers)
    
    # 0.0005  1 layer 1800 seed 41
    # 0.007 2 layer 100000 seed 41
    # 0.001 3 layer 2100 seed 41 
    history = network.fit(lrate, epochs, X_train, Y_train, X_valid, y_valid)
    loss = network.evalute(X_test, Y_test)
    
    return loss, history, network

if __name__ == '__main__':
    X_train, Y_train, X_valid, y_valid, X_test, Y_test, X, Y, labels = make_train_input_data()

    # print(X_train.shape)
    layers = config_layers(X_train)
    
    # network = SGDNetwork(layers=layers)
    
    epochs = 50000
    lrate = 0.05
    # SGD NETWORK
    # 0.0005  1 layer 1800 seed 41
    # 0.001 2 layers 1810 seed 41
    # 0.001 3 layers 2100 seed 41 
    
    # loss, history, network = calling_SGD(layers, X_train, Y_train, X_valid, y_valid, lrate, epochs)
    # history = network.fit(0.001, epochs, X_train, Y_train, X_valid, y_valid)
    # loss = network.evalute(X_test, Y_test)
    
    # BATCH NETWORK 
    loss, history, network = calling_BGD(layers, X_train, Y_train, X_valid, y_valid, lrate, epochs)
    # 2 layers learning rate 0.007 iteration: 100000 seed 41 
    # 1 layers learning rate 0.007 iteration: 50000 seed 41 
    # 3 layers learning rate 0.05 iteration: 50000 seed 41
    
    print("evaluation loss:", loss)
    whole_data = make_train_set([X, Y])
    whole_data = scaller_x.transform(
        whole_data.reshape(-1, 2)).reshape(whole_data.shape)

    train_pred = network.feed_forward(whole_data)
    # train_pred = network.feed_forward(X_train)
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(221, projection='3d')
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    network.ploting_errors(ax3, train_pred, labels)
    ax3.legend(loc="upper right")
    ax3.grid(True)

    train_pred = scaller_y.inverse_transform(train_pred.reshape(-1, 1))
    labels = scaller_y.inverse_transform(labels.reshape(-1, 1))

    best_epoch = network.ploting_learning_curves(ax1, ax2)
    ax2.grid(True)
    ax2.legend(loc="upper right")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax.set_title("prediction vs reality")
    ax.plot(X, Y, labels.ravel(), label="original")
    ax.plot(X, Y, train_pred.ravel(), label="prediction")

    ax.grid(True)
    ax.legend(loc="upper right")

    plt.grid(True)
    plt.show()
