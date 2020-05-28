from mnist import MNIST
import numpy as np
import random
import NeuralNet as nn

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

print(np_testing_Y)
print(np_testing_Y[0, :])

nn = nn.NeuralNetwork(np_testing_X[:5000, :], np_testing_Y[:5000, :], np_testing_X.shape[1], 32, 32, np_testing_Y.shape[1])

neural_network = nn.train_neural_network(np_testing_X[:5000], np_testing_Y[:5000, :])

random_index = random.randrange(0, 5000)

print(mndata.display(images[random_index]))

test = np.array(images[random_index]).reshape(((-1,1)))

print(labels[random_index])

answer = nn.predict(test)

for num in answer:
    print(num)


