import numpy as np
from sklearn.datasets import load_iris
import random


class NeuralNetwork:

    def __init__(self, X, Y, num_features, regularization=0.01, learning_rate=0.1, max_iterations=1000, *layers):
        """Sets up the hidden layers in the network and the initial values of theta."""

        # Define the number of layers and the size of each layer.
        self.number_of_layers = len(layers) + 1
        self.size_of_layers = list(layers)
        self.size_of_layers.insert(0, num_features)

        # Set up the theta and node lists.
        self.theta = []
        self.nodes = []

        # Set the number of features.
        self.num_features = num_features

        # Set the value of the attributes X and Y.
        self.X = X
        self.Y = Y

        # Set the level of regularization and the learning rate.
        self.regularization = regularization
        self.learning_rate = learning_rate

        # Define the maximum number of iterations to perform.
        self.max_iters = max_iterations

        # Create the theta and layer matrices
        self.initialise_nodes()

    def sigmoid_activation(self, np_array):
        """Apply sigmoid activation on the given layer."""

        # Need to prevent overflow error.
        signal = np.clip(np_array, -500, 500)
        return 1 / (1 + np.exp(-signal))

    def forward_propagation(self, X_inp):
        """Calculates the hypothesis via feed forward propagation."""

        # Insert the X_input into X
        self.nodes[0][1:, :] = X_inp

        # Perform vectorized feed forward propagation.
        for i in range(len(self.theta)):
            if i < len(self.theta) - 1:
                self.nodes[i + 1][1:, :] = np.matmul(self.theta[i], self.nodes[i])
                self.nodes[i + 1][1:, :] = self.sigmoid_activation(self.nodes[i + 1][1:, :])
            else:
                self.nodes[i + 1] = np.matmul(self.theta[i], self.nodes[i])
                self.nodes[i + 1] = self.sigmoid_activation(self.nodes[i + 1])

        return self.nodes[-1]

    def backward_propagation(self, hypothesis, Y_test, accumulator):
        """Perform backwards propagation which is used to train the neural network."""

        # Calculate the error between the hypothesis and the actual Y.
        last_error = hypothesis - Y_test.reshape((-1, 1))

        # Create a list of matrices which represent the error in each node.
        errors = [np.zeros(self.nodes[i].shape) for i in range(1, len(self.nodes) - 1)]
        # Append the error to the errors list.
        errors.append(last_error)

        # Now we need to actually perform the backwards propagation.
        for i in reversed(range(1, len(self.nodes) - 1)):
            # No bias unit in the final layer.
            if i == len(self.nodes) - 2:
                temp_error = np.matmul(self.theta[i].T, errors[i])
            else:
                temp_error = np.matmul(self.theta[i].T, errors[i][1:, :])

            # Need to multiply by the derivative of z.
            errors[i - 1] = temp_error * (self.nodes[i] * (1 - self.nodes[i]))

        # Now we have a list of all the errors.
        for i in range(len(accumulator)):
            # Update the accumulator. Remove the error of the bias node.
            if i == len(accumulator) - 1:
                accumulator[i] = accumulator[i] + np.matmul(errors[i], self.nodes[i].T)
            else:
                accumulator[i] = accumulator[i] + np.matmul(errors[i][1:, :], self.nodes[i].T)

        # Return the updated accumulator value.
        return accumulator

    def gradient_descent(self, delta):
        """This method performs gradient descent using the given delta."""
        for i in range(len(delta)):
            self.theta[i] = self.theta[i] - self.learning_rate * delta[i]

    def predict(self, X):
        """Returns the predictions for each class."""
        predictions = self.forward_propagation(X)
        return predictions

    def initialise_nodes(self):
        """Sets theta and the nodes in each layer."""

        # We're going to find epsilon for random initialisation
        epsilon = 0.5

        self.theta = []
        self.nodes = []  # We're also creating the hidden node matrices.

        # Append the X input nodes into the activation nodes.
        self.nodes.append(np.array([0 for i in range(self.num_features)], dtype='float').reshape((-1, 1)))
        # We are using a list and not a 3 dimensional matrix
        # so that we can store different number of nodes in different layers.
        self.nodes[0] = np.insert(self.nodes[0], 0, 1).reshape((-1, 1))

        # Calculate theta using random initialization and set each node to 0.
        for i in range(self.number_of_layers - 1):
            self.theta.append(np.random.rand(self.size_of_layers[i + 1],
                                             self.size_of_layers[i] + 1) * (2 * epsilon) - epsilon)
            self.nodes.append(np.zeros(self.size_of_layers[i + 1]))

            # Insert the bias node.
            if i != self.number_of_layers - 2:
                self.nodes[i + 1] = np.insert(self.nodes[i + 1], 0, 1).reshape((-1, 1))
            else:
                # Do not insert a bias node in the hypothesis
                self.nodes[i + 1] = self.nodes[i + 1].reshape((-1, 1))

    def train_neural_network(self):
        """Trains the neural network model."""

        # We need an accumulator.
        accumulator = [np.zeros((self.theta[i].shape[0], self.theta[i].shape[1])) \
                       for i in range(len(self.theta))]

        # Now we need to perform forward propagation.
        # Perform backward propagation on all the examples.
        # Do this for the specified number of iterations.
        for j in range(self.max_iters):
            print("Epoch: ", j+1)
            for i in range(self.X.shape[0]):
                hypothesis = self.forward_propagation(self.X[i, :].reshape((-1, 1)))
                accumulator = self.backward_propagation(hypothesis, self.Y[i, :], accumulator)

            m = self.X.shape[0]

            delta = [(1 / m) * accumulator[i] for i in range(len(self.theta))]

            # Add regularization. We do not regularize the bias term.
            for i in range(len(delta)):
                delta[i][:, 1:] = delta[i][:, 1:] + (self.regularization / m) * self.theta[i][:, 1:]

            self.gradient_descent(delta)

    def get_theta(self):
        return self.theta

    def save_theta(self):
        for i in range(len(self.theta)):
            np.savetxt('theta' + str(i) + '.csv', self.theta[i], delimiter=',')

    def load_theta(self, *files):
        self.theta.clear()
        for file in files:
            theta = np.loadtxt(file, delimiter=',')
            self.theta.append(theta)


def calc_accuracy(nn, X, Y):
    """Given a matrix of testing data and the corresponding labels the method returns the accuracy."""
    correct = 0
    for i in range(X.shape[0]):
        hypothesis = np.argmax(nn.predict(X[i].reshape((-1, 1))))
        if hypothesis == np.argmax(Y[i]):
            correct += 1
    return correct / X.shape[0]
