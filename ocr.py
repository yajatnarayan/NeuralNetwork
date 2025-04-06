import json
import math
import numpy as np
import os

class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    NUM_DIGITS = 10
    NN_FILE_PATH = 'nn.json'

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, train_indices, use_file=True):
        self._use_file = use_file
        self.num_hidden_nodes = num_hidden_nodes

        if use_file and os.path.exists(self.NN_FILE_PATH):
            self._load()
        else:
            # Initialize weights to small random values
            self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, self.NUM_DIGITS)
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weights(1, self.NUM_DIGITS)

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]

    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)

    def sigmoid(self, z):
        return np.vectorize(self._sigmoid_scalar)(z)

    def sigmoid_prime(self, z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def train(self, train_array):
        for data in train_array:
            # Forward propagation
            y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(data['y0']).T)
            sum1 = y1 + np.asmatrix(self.input_layer_bias)  # Add the bias
            y1 = self.sigmoid(sum1)

            y2 = np.dot(np.array(self.theta2), y1)
            y2 = np.add(y2, self.hidden_layer_bias)  # Add the bias
            y2 = self.sigmoid(y2)

            # Back propagation
            actual_vals = [0] * self.NUM_DIGITS
            actual_vals[data['label']] = 1
            output_errors = np.asmatrix(actual_vals).T - np.asmatrix(y2)
            hidden_errors = np.multiply(np.dot(np.asmatrix(self.theta2).T, output_errors), 
                                      self.sigmoid_prime(sum1))

            # Weight updates
            self.theta1 += self.LEARNING_RATE * np.dot(np.asmatrix(hidden_errors), 
                                                     np.asmatrix(data['y0']))
            self.theta2 += self.LEARNING_RATE * np.dot(np.asmatrix(output_errors), 
                                                     np.asmatrix(y1).T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(test).T)
        y1 = y1 + np.asmatrix(self.input_layer_bias)  # Add the bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)  # Add the bias
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": [np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
            "b1": self.input_layer_bias[0].tolist()[0],
            "b2": self.hidden_layer_bias[0].tolist()[0]
        }
        with open(self.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(self.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])] 