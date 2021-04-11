import numpy as np
import matplotlib.pyplot as plt
from neural_network.train import SGDNetwork
from neural_network.layer import make_layers, Layer
from neural_network.preprocess import (MIN_MAX_SCALAR, make_train_set,
                                        apply_k_fold, scaling)
from neural_network.utility_functions import fx


def make_train_input_data() -> tuple:
    X = np.linspace(0, 5, num=300) # feature 1
    Y = np.linspace(0, 5, num=300) # feature 2
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    labels = fx(X, Y)
    labels = (labels - np.min(labels.ravel())) / (np.max(labels.ravel()) - np.min(labels.ravel()))
    (training_set_samples) = 260

    X_train = make_train_set([X[:training_set_samples], Y[:training_set_samples]])
    Y_train = labels[:training_set_samples].reshape(1, -1)

    X_test = make_train_set([X[training_set_samples:], Y[training_set_samples:]])
    Y_test = labels[training_set_samples:].reshape(1, -1)
    
    
    features_shape = X_train.shape[-1::-1]
    X_train = X_train.reshape(*features_shape)
    X_test = X_test.reshape(X_test.shape[-1::-1])
    
    X_train, Y_train, X_valid, y_valid =  apply_k_fold(X_train, Y_train)
    
    return X_train, Y_train, X_valid, y_valid, X_test, Y_test


def config_layers(X_train) -> list:
    layers_config = [{
        Layer.NEURON: X_train.shape[0],
        Layer.INPUT_DIM: 0,
        Layer.LAYER_NUMBER: 0,
        Layer.IS_INPUT_LAYER: True
    },{
        Layer.NEURON: 16,
        Layer.INPUT_DIM: 2,
        Layer.LAYER_NUMBER: 1,
        Layer.ACTIVATION: "tanh",
        Layer.IS_INPUT_LAYER: False
    },
    # },{
    #     Layer.NEURON: 8,
    #     Layer.INPUT_DIM: 16,
    #     Layer.LAYER_NUMBER: 2,
    #     Layer.ACTIVATION: "relu",
    #     Layer.IS_INPUT_LAYER: False
    # },{
    #     Layer.NEURON: 4,
    #     Layer.INPUT_DIM: 8,
    #     Layer.LAYER_NUMBER: 3,
    #     Layer.ACTIVATION: "relu",
    #     Layer.IS_INPUT_LAYER: False
    # },
    {
        Layer.NEURON: 1,
        Layer.INPUT_DIM: 16,
        Layer.LAYER_NUMBER: 4,
        Layer.ACTIVATION: "linear",
        Layer.IS_INPUT_LAYER: False
    }]
    layers = make_layers(layers_config)
    
    return layers


if __name__ == '__main__':
    X_train, Y_train, X_valid, y_valid, X_test, Y_test = make_train_input_data()
    layers = config_layers(X_train)
    network = SGDNetwork(layers=layers)
    history = network.fit(0.00001, 1000, X_train, Y_train, X_valid, y_valid)
    loss = network.evalute(X_test, Y_test)
    
    print("evaluation loss:", loss)
    
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    