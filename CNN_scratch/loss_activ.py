import numpy as np


def one_hot_encode(y, num_classes=10):
    y = np.array(y, dtype='int')
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot


def shuffle_normalize(x, y, size):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    indices = indices[:size]
    x, y = x[indices], y[indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = one_hot_encode(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)


def leaky_relu_prime(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    return -np.sum(y_true * np.log(y_pred))


def categorical_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    return -1 * (y_true / y_pred)


def calculate_error(network, loss, X, Y):
    total_error = 0
    for x, y in zip(X, Y):
        output = network.predict(x)
        total_error += loss(y, output)
    average_error = total_error / len(X)
    return average_error
