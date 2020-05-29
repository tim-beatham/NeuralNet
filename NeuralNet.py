import numpy as np
from sklearn.datasets import load_iris
import random


class NeuralNetwork:

    def __init__(self, X, Y, num_features, *layers):
        """Sets up the hideen layers in the network and the inital values of theta."""

        self.number_of_layers = len(layers) + 1
        self.size_of_layers = list(layers)
        self.size_of_layers.insert(0, num_features)

        self.theta = []
        self.nodes = []

        self.num_features = num_features

        self.X = X
        self.Y = Y

        self.regularization = 0.01
        self.learning_rate = 0.0001

        self.max_iters = 1000

        self.initialise_nodes()

    def sigmoid_activation(self, np_array):
        """Apply sigmoid activation on the given layer."""

        # Need to prevent overflow error.
        signal = np.clip(np_array, -500, 500)

        return 1 / (1 + np.exp(-signal))

    def forward_propagation(self, X_inp):
        """Calcules the hypothesis via feed forward propagation."""

        # Insert the X_input into X

        self.nodes[0][1:, :] = X_inp

        for i in range(len(self.theta)):
            if i < len(self.theta) - 1:
                self.nodes[i + 1][1:, :] = np.matmul(self.theta[i], self.nodes[i])
                self.nodes[i + 1][1:, :] = self.sigmoid_activation(self.nodes[i + 1][1:, :])
            else:
                self.nodes[i + 1] = np.matmul(self.theta[i], self.nodes[i])
                self.nodes[i + 1] = self.sigmoid_activation(self.nodes[i + 1])

        return self.nodes[-1]

    def backward_propagation(self, hypothesis, Y_test, accumulator):
        """Perform backwards propagation which is used to trin the neural network."""

        # Going to do backwards propagation again.
        last_error = hypothesis - Y_test.reshape((-1, 1))

        # Create a list of the errors.
        errors = [np.zeros(self.nodes[i].shape) for i in range(1, len(self.nodes) - 1)]
        errors.append(last_error)

        # Now we need to actually perform the backwards propagation.
        for i in reversed(range(1, len(self.nodes) - 1)):
            # No bias unit on the final layer.
            if i == len(self.nodes) - 2:
                temp_error = np.matmul(self.theta[i].T, errors[i])
            else:
                temp_error = np.matmul(self.theta[i].T, errors[i][1:, :])

            errors[i - 1] = temp_error * (self.nodes[i] * (1 - self.nodes[i]))

        # Now we have a list of all the errors.
        for i in range(len(accumulator)):
            if i == len(accumulator) - 1:
                accumulator[i] = accumulator[i] + np.matmul(errors[i], self.nodes[i].T)
            else:
                accumulator[i] = accumulator[i] + np.matmul(errors[i][1:, :], self.nodes[i].T)

        return accumulator

    def gradient_descent(self, delt):
        """This method performs gradient descent using the given delt."""

        max_descent = 0
        prev_theta = self.theta

        for i in range(len(delt)):
            self.theta[i] = self.theta[i] - self.learning_rate * delt[i]
            difference = prev_theta[i] - self.theta[i]

            # Get the maximum
            if np.max(difference) > max_descent:
                max_descent = np.max(difference)

        # if max_descent < 10 ** -4:
        #   print("Converged!")
        #  return True

        return False

    def predict(self, X):
        predictions = self.forward_propagation(X)

        # Get the largest prediction

        return predictions

    def initialise_nodes(self):
        # We're going to find epsilon for random initialisation
        epsilon = 0.5

        self.theta = []
        self.nodes = []  # We're also creating the hidden node matrices.

        # Append the X input nodes into the activation nodes.
        self.nodes.append(np.array([0 for i in range(self.num_features)], dtype='float').reshape((-1, 1)))
        # We are using a list and not a 2 dimensional matrix as a two dimensional
        # so that we can store different number of nodes in different layers.
        self.nodes[0] = np.insert(self.nodes[0], 0, 1).reshape((-1, 1))

        for i in range(self.number_of_layers - 1):
            self.theta.append(np.random.rand(self.size_of_layers[i + 1],
                                             self.size_of_layers[i] + 1) * (2 * epsilon) - (epsilon))
            self.nodes.append(np.zeros(self.size_of_layers[i + 1]))

            if i != self.number_of_layers - 2:
                self.nodes[i + 1] = np.insert(self.nodes[i + 1], 0, 1).reshape((-1, 1))
                # Do not want to insert a 1 in the last layer
            else:
                self.nodes[i + 1] = self.nodes[i + 1].reshape((-1, 1))

    def train_neural_network(self, X, Y):
        """Trains the neural network model.
        Layers passed in contains two keys,
        number of layers and array containing
        the size of each layer."""

        # Now we need to instantiate the layers.
        # We need to use random initialization this time.
        # Zero initialization will not work.

        # We need an accumulator.
        accumulator = [np.zeros((self.theta[i].shape[0], self.theta[i].shape[1])) \
                       for i in range(len(self.theta))]

        # Now we need to perform forward propagation.
        # Perform backward propagatio on all the examples.
        for j in range(self.max_iters):
            for i in range(X.shape[0]):
                hypothesis = self.forward_propagation(X[i, :].reshape((-1, 1)))
                accumulator = self.backward_propagation(hypothesis, Y[i, :], accumulator)

            m = X.shape[0]

            delt = [(1 / m) * accumulator[i] for i in range(len(self.theta))]

            # Add regularization. We do not regularize the bias term.
            for i in range(len(delt)):
                delt[i][:, 1:] = delt[i][:, 1:] + (self.regularization / m) * self.theta[i][:, 1:]

            done = self.gradient_descent(delt)

            if j == self.max_iters - 1:
                print("The maximum number of iterations has been reached!")

            if done:
                print("Converged!")
                break

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


"""
X = []
Y = []

for i in range(1000):
    if i % 2 == 0:
        X.append([1])
        Y.append([0])
    else:
        X.append([2])
        Y.append([1])

X = np.array(X, dtype='float')
Y = np.array(Y, dtype='float')

# l and sizes must match
nn = NeuralNetwork(X, Y, X.shape[1], 1, 1, Y.shape[1])
nn.train_neural_network(X, Y)

example = np.array([[1]])

print(np.where(nn.predict(np.array([2])) > 0.5, 1, 0))

iris = load_iris()
print(iris.data.shape)
print(iris.target.reshape((-1, 1)).shape)

X = iris.data
Y = iris.target

# Transform Y into the required format.
Y_iris = np.zeros((Y.shape[0], 3))

for i in range(Y_iris.shape[0]):
    Y_iris[i, Y[i]] = 1

# l and sizes must match
nn = NeuralNetwork(X, Y_iris, X.shape[1], 30, 30, Y_iris.shape[1])
nn.train_neural_network(X, Y_iris)

correct = 0

print(Y_iris)

for i in range(X.shape[0]):
    # Make a prediction.
    predictions = nn.predict(X[i].reshape((-1, 1)))
    maximum = np.argmax(predictions)

    print(maximum)

    if maximum == Y[i]:
        correct += 1

print("Accuracy: ", correct / X.shape[0])

nn.save_theta()
"""
