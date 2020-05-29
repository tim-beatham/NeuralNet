import numpy as np
from sklearn.datasets import load_iris
from NeuralNet import NeuralNetwork
from NeuralNet import calc_accuracy
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data
Y = iris.target

# Transform Y into the required format.
Y_iris = np.zeros((Y.shape[0], 3))

for i in range(Y_iris.shape[0]):
    Y_iris[i, Y[i]] = 1

# Split the data into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, Y_iris, random_state=76)

nn = NeuralNetwork(X_train, y_train, X_train.shape[1], 0.01, 0.1, 1000, 32, y_train.shape[1])
nn.train_neural_network()

print("Training accuracy: " + str(calc_accuracy(nn, X_train, y_train)))
print("Testing accuracy: " + str(calc_accuracy(nn, X_test, y_test)))

nn.save_theta()
