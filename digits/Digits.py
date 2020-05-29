from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import numpy as np
from NeuralNet import NeuralNetwork

digits = load_digits()

X = digits.data
Y = digits.target

Y_classes = np.zeros((X.shape[0], 10))

for i in range(Y.shape[0]):
    Y_classes[i, Y[i]] = 1

Y = Y_classes

X_train, X_test, y_train, y_test = train_test_split(X, Y)
nn = NeuralNetwork(X_train, y_train, X_train.shape[1], 0.01, 0.1, 1000, 100, 100, y_train.shape[1])
# nn.train_neural_network()

# Save theta values.
# nn.save_theta()


# This is executed once we have trained the neural network.
index = random.randrange(0, X_test.shape[0])

nn.load_theta('theta0.csv', 'theta1.csv', 'theta2.csv')

prediction = np.argmax(nn.predict(X_test[index, :].reshape((-1,1))))
label = np.argmax(y_test[index])

plt.gray()
plt.matshow(X_test[index, :].reshape((8,8)))
plt.xlabel("Prediction: " + str(prediction) + " Label: " + str(label))
plt.show()





