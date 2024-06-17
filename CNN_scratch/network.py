from keras.datasets import mnist
from loss_activ import *
from scipy import signal


class LinearLayer:
    def __init__(self, input_size, output_size, activation=sigmoid, activation_prime=sigmoid_prime):
        self.input = None
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)  # Xavier init.
        self.bias = np.random.randn(output_size, 1)
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, x):
        self.input = x
        linear_output = self.weights @ self.input + self.bias
        if self.activation:
            return self.activation(linear_output)
        else:
            return linear_output

    def backward(self, output_gradient, learning_rate):
        if self.activation:
            act_prime = self.activation_prime(self.weights @ self.input + self.bias)
            output_gradient = np.multiply(output_gradient, act_prime)
        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.weights.T @ output_gradient

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class ConvolutionalLayer:
    def __init__(self, input_shape, kernel_size, depth, activation=leaky_relu, activation_prime=leaky_relu_prime):
        self.output = None
        self.input = None

        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        # He init.
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2 / np.prod(self.kernels_shape[1:]))

        self.biases = np.random.randn(*self.output_shape)
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, x):
        self.input = x
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        if self.activation:
            self.output = self.activation(self.output)

        if np.any(np.isnan(self.output)):
            raise ValueError("Array contains NaN values.")

        return self.output

    def backward(self, output_gradient, learning_rate):
        output_gradient = np.multiply(output_gradient, self.activation_prime(self.output))

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.convolve2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        return np.reshape(x, self.output_shape)

    def backward(self, output_gradient, _):
        return np.reshape(output_gradient, self.input_shape)


class Network:
    def __init__(self, loss, loss_prime, epochs, step, batch_size=5):
        self.layers_list = []
        self.loss = loss
        self.loss_prime = loss_prime
        self.epochs = epochs
        self.step = step
        self.batch_size = batch_size

    def specify_layers(self, layers_list):
        self.layers_list = layers_list

    def fit(self, X, Y):
        for e in range(self.epochs):
            error = 0
            for x, y in zip(X, Y):
                output = self.predict(x)
                error += self.loss(y, output)

                grad = self.loss_prime(y, output)
                for layer in reversed(self.layers_list):
                    grad = layer.backward(grad, self.step)

            error /= len(x_train)
            print(f"{e + 1}/{self.epochs}, error={error}")

    def predict(self, X):
        output = X
        for layer in self.layers_list:
            output = layer.forward(output)
        return output


# load MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = shuffle_normalize(x_train, y_train, 1000)
x_test, y_test = shuffle_normalize(x_test, y_test, 1000)

nn = Network(
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    5,
    0.01
)

nn.specify_layers(
    [
        ConvolutionalLayer((1, 28, 28), 3, 8),
        Reshape((8, 26, 26), (8 * 26 * 26, 1)),
        LinearLayer(8 * 26 * 26, 32, leaky_relu, leaky_relu_prime),
        LinearLayer(32, 10)
    ]
)

nn.fit(x_train, y_train)

test_error = calculate_error(nn, categorical_cross_entropy, x_test, y_test)
print(f"Test error: {round(test_error, 5)}")
