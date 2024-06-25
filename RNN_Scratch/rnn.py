import numpy as np
import matplotlib.pyplot as plt


class RNN:
    def __init__(self, input_size, hidden_units, output_size, learning_rate=0.01):
        self.hidden_units = hidden_units
        self.lr = learning_rate
        # Xavier init.
        self.Wx = np.random.randn(hidden_units, input_size) * np.sqrt(2.0 / (hidden_units + input_size))
        self.Wh = np.random.randn(hidden_units, hidden_units) * np.sqrt(2.0 / (hidden_units + hidden_units))
        self.Wy = np.random.randn(output_size, hidden_units) * np.sqrt(2.0 / (output_size + hidden_units))

    def forward(self, sample_x):
        ht = np.zeros((self.hidden_units, 1))
        hidden_states = [ht]
        inputs = []

        yt = None
        for step in range(len(sample_x)):
            h_t = np.tanh(np.dot(self.Wx, sample_x[step].reshape(-1, 1)) + np.dot(self.Wh, ht))
            yt = np.dot(self.Wy, h_t)
            inputs.append(sample_x[step].reshape(-1, 1))
            hidden_states.append(h_t)

        return yt, hidden_states, inputs

    def backward(self, yt, sample_y, hidden_states, inputs):
        error = yt - sample_y
        loss = 0.5 * error ** 2

        n = len(inputs)
        dyt = error
        dWy = np.dot(dyt, hidden_states[-1].T)
        dht = np.dot(dyt, self.Wy).T
        dWx = np.zeros(self.Wx.shape)
        dWh = np.zeros(self.Wh.shape)

        for step in reversed(range(n)):
            temp = (1 - hidden_states[step + 1] ** 2) * dht
            dWx += temp @ inputs[step].T
            dWh += temp @ hidden_states[step].T
            dht = self.Wh @ temp

        dWy = np.clip(dWy, -1, 1)
        dWx = np.clip(dWx, -1, 1)
        dWh = np.clip(dWh, -1, 1)

        self.Wy -= self.lr * dWy
        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh

        return loss

    def fit(self, x, y, epochs):
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample in range(x.shape[0]):
                sample_x, sample_y = x[sample], y[sample]
                yt, hidden_states, inputs = self.forward(sample_x)
                loss = self.backward(yt, sample_y, hidden_states, inputs)
                epoch_loss += loss
            epoch_loss /= x.shape[0]
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

    def predict(self, x):
        outputs = []
        for sample in range(len(x)):
            yt, _, _ = self.forward(x[sample])
            outputs.append(yt)
        return np.array(outputs).reshape(-1, 1)


def generate_data(start=0, end=10, step=0.1, timesteps=25, noise_level=0.05):
    x, y = [], []
    t = np.arange(start, end, step)
    sin_wave = np.sin(t)
    noise = np.random.normal(0, noise_level, sin_wave.shape)
    noisy_sin_wave = sin_wave + noise
    for i in range(noisy_sin_wave.shape[0] - timesteps):
        x.append(noisy_sin_wave[i:i + timesteps])
        y.append(noisy_sin_wave[i + timesteps])
    return np.array(x).reshape(len(y), timesteps, 1), np.array(y).reshape(len(y), 1), t


x_train, y_train, t_train = generate_data(0, 10)
x_test, y_test, t_test = generate_data(0, 15)

rnn = RNN(input_size=1, hidden_units=100, output_size=1)
rnn.fit(x_train, y_train, 15)

pred = rnn.predict(x_test)

plt.figure(figsize=(12, 6), dpi=120)
plt.scatter(t_test[25:], y_test, label='True values', color='blue')
plt.plot(t_test[25:], pred, label='Predictions', color='red')
plt.axvline(x=10, color='black', linestyle='--', label='Split')
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.legend()
plt.title('Sine Wave Prediction')
plt.show()
