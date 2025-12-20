import json
import math
import numpy as np
import os


class OCRNeuralNetwork:
    """Simple 3-layer neural network for 20x20 pixel digit recognition.

    Internal shapes:
      - input layer size: 400 (20x20)
      - hidden layer size: H (configurable)
      - output layer size: 10

    We store weights as numpy arrays with the following shapes:
      - theta1: (H, 400)
      - b1: (H, 1)
      - theta2: (10, H)
      - b2: (10, 1)
    """

    LEARNING_RATE = 0.1
    NUM_DIGITS = 10
    INPUT_SIZE = 400
    NN_FILE_PATH = 'nn.json'

    def __init__(self, num_hidden_nodes, data_matrix=None, data_labels=None, train_indices=None, use_file=True):
        self._use_file = use_file
        self.num_hidden_nodes = int(num_hidden_nodes)

        if use_file and os.path.exists(self.NN_FILE_PATH):
            self._load()
        else:
            # Initialize weights to small random values in correct shapes
            self.theta1 = (np.random.rand(self.num_hidden_nodes, self.INPUT_SIZE) * 0.12) - 0.06
            self.theta2 = (np.random.rand(self.NUM_DIGITS, self.num_hidden_nodes) * 0.12) - 0.06
            self.b1 = np.zeros((self.num_hidden_nodes, 1))
            self.b2 = np.zeros((self.NUM_DIGITS, 1))

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def train(self, train_array):
        """Train on a list of samples. Each item should be a dict with keys:
           - 'y0': list/array of length 400 (input features)
           - 'label': integer 0-9
        Updates weights in-place using simple per-sample gradient descent.
        """
        for data in train_array:
            x = np.array(data['y0'], dtype=float).reshape((self.INPUT_SIZE, 1))

            # Forward pass
            z1 = self.theta1.dot(x) + self.b1
            a1 = self._sigmoid(z1)

            z2 = self.theta2.dot(a1) + self.b2
            a2 = self._sigmoid(z2)

            # Prepare target
            y = np.zeros((self.NUM_DIGITS, 1))
            lbl = int(data['label'])
            y[lbl, 0] = 1.0

            # Backprop
            delta2 = a2 - y  # (10,1)
            delta1 = self.theta2.T.dot(delta2) * self._sigmoid_prime(z1)  # (H,1)

            # Gradient descent parameter update (simple online update)
            self.theta2 -= self.LEARNING_RATE * (delta2.dot(a1.T))
            self.b2 -= self.LEARNING_RATE * delta2

            self.theta1 -= self.LEARNING_RATE * (delta1.dot(x.T))
            self.b1 -= self.LEARNING_RATE * delta1

    def predict(self, test, return_activations=False):
        """Predict the digit for a given test input (list or array of length 400).
        Returns the integer digit with highest output activation.
        If return_activations=True, returns a dict with prediction and all layer activations.
        """
        x = np.array(test, dtype=float).reshape((self.INPUT_SIZE, 1))
        z1 = self.theta1.dot(x) + self.b1
        a1 = self._sigmoid(z1)
        z2 = self.theta2.dot(a1) + self.b2
        a2 = self._sigmoid(z2)

        prediction = int(np.argmax(a2))

        if return_activations:
            return {
                'prediction': prediction,
                'input_vector': x.flatten().tolist(),
                'hidden_activations': a1.flatten().tolist(),
                'output_activations': a2.flatten().tolist()
            }

        return prediction

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            'theta1': self.theta1.tolist(),
            'theta2': self.theta2.tolist(),
            'b1': self.b1.tolist(),
            'b2': self.b2.tolist()
        }
        with open(self.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(self.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = np.array(nn['theta1'], dtype=float)
        self.theta2 = np.array(nn['theta2'], dtype=float)
        self.b1 = np.array(nn['b1'], dtype=float)
        self.b2 = np.array(nn['b2'], dtype=float)