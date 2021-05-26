from ..activation import activation_detection, initilizing
import numpy as np
from functools import wraps
# Xavier/Glorot Initialization: None, hyperbolic Tan (tanh), Logistic(sigmoid), softmax.
# He Initialization: Rectified Linear activation unit(ReLU) and Variants.
# bring the variance of those outputs to approximately one
# gradient vanishing and exploiding

class Layer:
    NEURON, INPUT_DIM, LAYER_NUMBER = "neuron", "input_dim", "layer_number"
    IS_INPUT_LAYER, ACTIVATION = "is_input_layer", "activition"
    VDB, VDW, WEIGHT, BIAS = "VDB", "VDW", "WEIGHT", "BIAS"
    FREEZ, NAME, VALUE = "freez", "name", "value"
    
    def __init__(self, neuron: int, input_dim: int, layer_number: int,
                 activation: str = None, is_input_layer: bool = False,
                 layer_name: str = "Dense") -> None:
        self.neuron = neuron
        self.input_dim = input_dim
        self.layer_number = layer_number
        self.activation_function = activation
        self.activation_setup()
        self.layer_name = layer_name + " " + str(layer_number)
        self.net, self.O, self.delta = None, None, None
    
        if not is_input_layer or layer_number != 0:
            self.VDW = np.zeros(shape=(self.neuron, input_dim))
            self.VDB = np.zeros(shape=(self.neuron, 1))
            if activation == "sigmoid" or activation == "tanh": # glorot
                self.W, self.B = initilizing("Glorot", self.neuron, input_dim)
            elif activation == "relu" or activation == "linear": # he normal
                self.W, self.B = initilizing("he_normal", self.neuron, input_dim)
            self.trainable_param_maker = lambda f, name, value: {"freez": f, "name": name, "value": value}
            self.trainable_params = [
                self.trainable_param_maker(False, "VDW", self.VDW),
                self.trainable_param_maker(False, "VDB", self.VDB),
                self.trainable_param_maker(False, "WEIGHT", self.W),
                self.trainable_param_maker(False, "BIAS", self.B)
            ]
        self.__layers_is_freezed = False
    
    def __repr__(self) -> str:
        repres_list = [self.neuron, self.input_dim, self.activation, self.layer_name]
        if self.layer_number != 0:
            repres_list = repres_list + [self.W.shape, self.B.shape]
        else:
            repres_list = repres_list + [None, None]
        repres_list.append(self.layer_name)
        return """ neurons: {} \t\t input_dim: {} \t\t activation: {} \n
                   \r layer_name: {} \t\t weights_dimentions: {} \t\t bias_dimentions: {} \n
                   \r _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _{} \r\n
                   """.format(*repres_list)
    def __str__(self) -> str:
        return self.__repr__()

    def unfreez_the_layer(self):
        self.__layers_is_freezed = False
        
    def activation_setup(self):
        if self.activation_function is not None:
            self.activation_func = activation_detection(self.activation_function)
        
    def activation(self, input):
        input = np.asarray(input, np.float128)
        return self.activation_func(input)
    
    def freez_the_layer(self):
        self.__layers_is_freezed = True
        
    def is_freezed(self) -> bool :
        return self.__layers_is_freezed

    def is_parameter_freez(self, value) -> bool:
        for parameter in self.trainable_params:
            if np.allclose(parameter[Layer.VALUE], value):
                return parameter[Layer.FREEZ]
        
        raise Exception(f"Parameter does not exits in layer number: {self.layer_number}")
    
def config_wrapper(func):
    @wraps(func)
    def checker(*args, **kwargs):
        check_config_list = [Layer.INPUT_DIM, Layer.NEURON, Layer.LAYER_NUMBER, Layer.IS_INPUT_LAYER, Layer.ACTIVATION]
        configs = args[0]
        for config in configs:
            if not isinstance(config, dict):
                raise Exception("Config of layer must be dictionary")
            for key in config.keys():
                if key not in check_config_list:
                    print(key)
                    raise Exception(
                        "congif of layers must have Neuron number and input Dim and layer number"
                    )
        return func(*args, **kwargs)
    return checker

@config_wrapper           
def make_layers(layers_config: list) -> list:
    layers = []
    for config in layers_config:            
        l = Layer(
                neuron=int(config[Layer.NEURON]),
                input_dim=int(config[Layer.INPUT_DIM]),
                layer_number=int(config[Layer.LAYER_NUMBER]),
                activation=config.get(Layer.ACTIVATION, None),
                is_input_layer=config.get(Layer.IS_INPUT_LAYER, False)
            )
        layers.append(l)
        
    return layers