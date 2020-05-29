from mnist import MNIST
import numpy as np
import random
import NeuralNet
from sklearn.model_selection import train_test_split

mndata = MNIST('samples')

images, labels = mndata.load_training()

mntesting = MNIST('testing')

image_test, image_label = mntesting.load_testing()

# nn = NeuralNet(images, labels, images.shape[1], 100, 100, labels.shape[1])

# nn.train_neural_network(images, labels)

np_testing_X = np.array([images[i] for i in range(len(images))])
np_testing_Y = np.zeros((len(image_label), 10))

for i in range(np_testing_Y.shape[0]):
    index = image_label[i]
    np_testing_Y[i, index] = 1


nn = NeuralNet.NeuralNetwork(np_testing_X, np_testing_Y, np_testing_X.shape[1], 800, np_testing_Y.shape[1])

# nn.train_neural_network(X_test, y_test)

# Just gonna save the theta in a file.
# nn.save_theta()

nn.load_theta("theta0.csv", "theta1.csv")


# Pick a random index.

index = random.randrange(0, len(np_testing_X))

print(mndata.display(np_testing_X[index]))

# Make a prediction
print("Prediction:", nn.predict(np_testing_X[index].reshape((-1,1))))
