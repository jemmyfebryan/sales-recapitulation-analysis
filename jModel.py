import numpy as np
import pickle

class Converter_runtime:
    def __init__(self, layers, weights, hyperparameter_separator):
        self.layers = layers
        self.weights = weights
        self.hyperparameter_separator = hyperparameter_separator

class Model:
    version = "0.1.0"

    def __init__(self):
        self.layers = []

    def load_model(self, file_path):
        if file_path[-7:] != ".jmodel": raise NameError("Model should ended with .jmodel")
        with open(file_path, 'rb') as file:
            converter = pickle.load(file)
        for ind, layer in enumerate(converter.layers):
            components = layer.split(sep=converter.hyperparameter_separator)
            exec(f"self.layers.append({components[0]}(weights=converter.weights[ind],{','.join(components[1:])}))")

    def predict(self, input_array):
        for layer in self.layers:
            input_array = layer.feed(input_array)
        return input_array

class Activation:
    @staticmethod
    def linear(input_array):
        return input_array
    
    @staticmethod
    def relu(input_array):
        return np.maximum(0, input_array)
    
    @staticmethod
    def sigmoid(input_array):
        return 1 / (1 + np.exp(-input_array))
    
    @staticmethod
    def tanh(input_array):
        return np.tanh(input_array)
    
    @staticmethod
    def softmax(input_array, axis=-1):
        exp_values = np.exp(input_array - np.max(input_array, axis=axis, keepdims=True))
        return exp_values / np.sum(exp_values, axis=axis, keepdims=True)

class Reshape:
    def __init__(self, weights, target_shape):
        self.target_shape = target_shape

    def feed(self, input_array):
        return input_array.reshape(self.target_shape)
    
class Flatten:
    def __init__(self, weights):
        pass

    def feed(self, input_array):
        return input_array.flatten()


class Dense:
    def __init__(self, weights, use_bias=True, activation='linear'):
        self.weights = weights
        self.activation = getattr(Activation, activation)
        self.use_bias = use_bias

    def feed(self, input_array):
        output = np.dot(input_array, self.weights[0])
        if self.use_bias:
            output += self.weights[1]
        return self.activation(output)
    
class LSTM:
    def __init__(self, weights, units, activation='tanh', use_bias=True, return_sequences=False):
        self.weights = weights
        self.units = int(units)
        self.activation = getattr(Activation, activation)
        self.use_bias = use_bias
        self.return_sequences = return_sequences

    def feed(self, input_array):
        # Initialize weights and biases
        hidden_dim = self.units

        # Input-to-hidden weights
        w_ih = self.weights[0]
        # Hidden-to-hidden weights
        w_hh = self.weights[1]
        # Biases
        if not self.use_bias:
            b = np.zeros(4 * hidden_dim)
        else:
            b = self.weights[2]

        seq_len = input_array.shape[1]

        hiddens = []
        c = np.zeros((input_array.shape[0], hidden_dim))  # Cell state initialization
        h = np.zeros((input_array.shape[0], hidden_dim))  # Hidden state initialization

        for t in range(seq_len):
            x_t = input_array[:, t, :]

            # LSTM cell operations
            gates = np.dot(x_t, w_ih) + np.dot(h, w_hh) + b
            input_gate = Activation.sigmoid(gates[:, :hidden_dim])
            forget_gate = Activation.sigmoid(gates[:, hidden_dim:2*hidden_dim])
            candidate_cell = self.activation(gates[:, 2*hidden_dim:3*hidden_dim])
            output_gate = Activation.sigmoid(gates[:, 3*hidden_dim:])

            c = forget_gate * c + input_gate * candidate_cell
            h = output_gate * self.activation(c)

            hiddens.append(h)

        if self.return_sequences:
            output = np.array(hiddens).transpose(1, 0, 2)
        else:
            output = h

        return output