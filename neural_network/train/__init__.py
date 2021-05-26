import numpy as np
from ..activation import activation_detection, drivative_of_activation
from ..metrics import get_loss_func
from ..scheduling import learning_scheduling
from ..utility_functions import progress_bar


class Network:
    VALIDATION_LOSS = "val_loss"
    TRAIN_LOSS = "train_loss"
    WEIGHT_HOLDER = "WEIGHT_HOLDER"
    
    def __init__(self, layers: list) -> None:
        self.history = {
            Network.VALIDATION_LOSS: [],
            Network.TRAIN_LOSS: [],
            Network.WEIGHT_HOLDER: []
        }
        self.layers = layers
        self.epochs = 1000
        
    def feed_forward(self, feature_inputs):
        self.layers[0].net = feature_inputs
        self.layers[0].O = feature_inputs
        for layer_index in range(1, len(self.layers)):
            self.layers[layer_index].net = np.dot(self.layers[layer_index].W, self.layers[layer_index - 1].O) + self.layers[layer_index].B
            self.layers[layer_index].O = self.layers[layer_index].activation(self.layers[layer_index].net)
        return self.layers[-1].O

# assert (feed_forward(X_train).shape == Y_train.shape)
    def back_propagation(self, y_pred, y_true, is_batch = False):
        error = y_pred - y_true # for get rid of -1
        delta = error # last layer delta and error and drivative is one(f'(net) = 1)
        for i in range(len(self.layers) - 1, 0, -1):
            if i != len(self.layers) - 1:
                delta = np.dot(self.layers[i+1].W.T, delta) * drivative_of_activation(self.layers[i].net, self.layers[i].activation_function)

            if not is_batch:
                assert delta.shape[0] == self.layers[i].neuron
            
                self.layers[i].delta = delta
                self.layers[i].dw = np.dot(delta, self.layers[i - 1].O.T)
                self.layers[i].db = delta
            else:
                self.layers[i].delta = delta
                self.layers[i].dw = np.dot(delta, self.layers[i - 1].O.T) / y_pred.shape[-1]
                self.layers[i].db = np.sum(delta, axis=1, keepdims=True) / y_pred.shape[-1]
                # print(delta.shape, self.layers[i].db.shape)
                # import sys
                # sys.exit(1)
            
            assert self.layers[i].dw.shape == self.layers[i].W.shape
            assert self.layers[i].db.shape == self.layers[i].B.shape


    def update_weights(self, lrate, optimizer, ß = 0.9):
        weight_layer = []
        for layer in self.layers[1:]:
            weight_layer.append((layer.W, layer.B))
            if optimizer == "momentum":
                if not layer.is_freezed():
                    layer.VDW = ß * layer.VDW + (1-ß) * layer.dw
                    layer.VDB = ß * layer.VDB + (1-ß) * layer.db
                    layer.W = layer.W - lrate * layer.VDW
                    layer.B = layer.B - lrate * layer.VDB
            elif optimizer == "sgd" or optimizer == "batch":
                if not layer.is_freezed():
                    # if not layer.is_parameter_freez(layer.W):
                    layer.W = layer.W - lrate * layer.dw
                    # if not layer.is_parameter_freez(layer.B):
                    layer.B = layer.B - lrate * layer.db
            else:
                raise Exception(f"optimizer: {optimizer} is not supported")
        
            self.history[Network.WEIGHT_HOLDER].append(weight_layer)

    def fit(self, lrate: float, epochs: int,
        X_train: np.ndarray, y_train: np.ndarray,
        X_valid: np.ndarray, y_valid: np.ndarray,
        cost_func: str = "mse",
        learning_decay: bool = False, tolorance: float=0.05) -> dict:
                
        self.history = {
            Network.VALIDATION_LOSS: [],
            Network.TRAIN_LOSS: [],
            Network.WEIGHT_HOLDER: []
        }
        
        self.train(lrate, epochs, X_train, y_train,
                    X_valid, y_valid, cost_func,
                   learning_decay, tolorance)
                
        return self.history
        
    def train(lrate: float, epochs: int,
            X_train: np.ndarray, y_train: np.ndarray,
            X_valid: np.ndarray, y_valid: np.ndarray,
            cost_func: str = "mse",
            learning_decay: bool = False, tolorance: float=0.05):
        raise NotImplementedError
    
    def evalute(self, X_test: np.ndarray, Y_label: np.ndarray, mean=False):
        Y_pred = self.feed_forward(X_test)
        loss = get_loss_func(Y_pred, Y_label, mean=mean)
        self.history["loss"] = loss
        return loss

    def ploting_learning_curves(self, ax, ax1):
        ax.set_title("learning_curves")
        
        validation_losses = np.array(self.history[Network.VALIDATION_LOSS])
        best_epoch = np.argmin(validation_losses.ravel()[50:]) + 50
        
        test_point = self.history[Network.VALIDATION_LOSS][best_epoch]
        
        arrow_prop = dict(
            arrowstyle = "->",
            connectionstyle = "angle, angleA = 0, angleB = 90, rad = 10"
        )
        ax.annotate(
            text= "early stopping" if best_epoch in range(self.epochs-10, self.epochs+2) else "maybe overfiting occured",
            xy=(best_epoch, test_point),
            xytext=(best_epoch - 40, np.max(validation_losses[100:].ravel()) - np.max(validation_losses[100:].ravel()) * 0.1),
            arrowprops = arrow_prop,
            bbox = dict(boxstyle ="round", fc ="0.8")
        )
        ax.plot(self.history[Network.TRAIN_LOSS][100:], label="traning_loss")
        ax.plot(self.history[Network.VALIDATION_LOSS][100:], label="validation_loss")
        ax1.plot(self.history[Network.VALIDATION_LOSS][10:], color='g',
                 marker=".", label="validation_loss", linewidth=0.5)
        # top_val = np.max(self.history[Network.VALIDATION_LOSS])
        # ax.set_ylim([0, top_val if top_val > 1 else top_val * 2 ])
        return best_epoch        
    
    def ploting_errors(self, ax, prediction, y_true):
        errors = np.abs(prediction.ravel() - y_true.ravel())
        ax.plot(errors, color="r", marker=".", label="error of data test_set", linewidth=0.5)
        # ax.set_ylim(-1, 1)
        
            

class SGDNetwork(Network):
    def __init__(self, layers: list) -> None:
        super().__init__(layers)
        self.optimizer = "sgd"
    
    def train(self, lrate: float, epochs: int,
            X_train: np.ndarray, y_train: np.ndarray,
            X_valid: np.ndarray, y_valid: np.ndarray,
            cost_func: str = "mse",
            learning_decay: bool = False, tolorance: float = 0.05):
        
        self.epochs = epochs
        eta = lrate
        for epoch in range(1, epochs+1):
                    
            if learning_decay:
                eta = learning_scheduling(lrate, epoch)
                # _, _, X_valid, y_valid =  apply_k_fold(X_train, y_train)
                    
            for i in range(X_train.shape[-1]):
                sample = X_train[:, i].reshape(-1, 1)
                y_pred = self.feed_forward(sample)
                assert y_pred.shape == y_train[:, i].reshape(-1, 1).shape
                self.back_propagation(y_pred, y_train[:, i].reshape(-1, 1))
                self.update_weights(eta, optimizer=self.optimizer)
            
            print(progress_bar(epoch, epochs))
                    
            train_pred = self.feed_forward(X_train)
            train_loss = get_loss_func(train_pred, y_train, mean=False)
            self.history[Network.TRAIN_LOSS].append(train_loss)
        
            valid_pred = self.feed_forward(X_valid)
            validation_loss = get_loss_func(valid_pred, y_valid, mean=False)
            self.history[Network.VALIDATION_LOSS].append(validation_loss)
            print("train_loss: %.6f \t validation_loss: %.6f" % (train_loss, validation_loss))
        
            # if validation_loss < tolorance:
            #     break
    
class BatchNetwork(Network):
    def __init__(self, layers: list) -> None:
        super().__init__(layers)
        self.optimizer = "batch"
    
    def train(self, lrate: float, epochs: int,
                X_train: np.ndarray, y_train: np.ndarray,
                X_valid: np.ndarray, y_valid: np.ndarray,
                cost_func: str = "mse",
                learning_decay: bool = False, tolorance: float = 0.05):

        eta = lrate
        for epoch in range(epochs):
            if learning_decay:
                eta = learning_scheduling(lrate, epoch)
    
            y_hats = self.feed_forward(X_train)
            assert y_hats.shape == y_train.shape
            self.back_propagation(y_hats, y_train, is_batch=True)
            self.update_weights(eta, optimizer=self.optimizer)
            
            train_pred = self.feed_forward(X_train)
            train_loss = get_loss_func(train_pred, y_train, mean=False)
            self.history[Network.TRAIN_LOSS].append(train_loss)
                
            valid_pred = self.feed_forward(X_valid)
            test_loss = get_loss_func(valid_pred, y_valid, mean=False)
            self.history[Network.VALIDATION_LOSS].append(test_loss)
            if epoch %100 == 0:
                print(progress_bar(epoch, epochs))
                print("train_loss: %.6f \t validation_loss: %.6f" % (train_loss, test_loss))