from tqdm import tqdm
import numpy as np


def xavier_init(input_size, output_size):
    return np.random.uniform(-1, 1,
                             (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))


def sigmoid(input, derivative=False):
    if derivative:
        return input * (1 - input)
    return 1 / (1 + np.exp(-input))


def tanh(input, derivative=False):
    if derivative:
        return 1 - input ** 2
    return np.tanh(input)


def softmax(input):
    exp_input = np.exp(input - np.max(input))
    return exp_input / np.sum(exp_input, axis=0)


def one_hot(text):
    output = np.zeros((unique_chars, 1))
    output[char_to_idx[text]] = 1

    return output


class LSTM:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        self.wf = xavier_init(input_size, hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        self.wi = xavier_init(input_size, hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        self.wc = xavier_init(input_size, hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        self.wo = xavier_init(input_size, hidden_size)
        self.bo = np.zeros((hidden_size, 1))

        self.wh = xavier_init(hidden_size, output_size)
        self.bh = np.zeros((output_size, 1))

        self.input_gates = None
        self.outputs = None
        self.forget_gates = None
        self.output_gates = None
        self.candidate_gates = None
        self.activation_outputs = None
        self.cell_states = None
        self.hidden_states = None
        self.concat_inputs = None

    def forward(self, inputs):
        self.concat_inputs = {}

        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1: np.zeros((self.hidden_size, 1))}

        self.activation_outputs = {}
        self.candidate_gates = {}
        self.output_gates = {}
        self.forget_gates = {}
        self.input_gates = {}
        self.outputs = {}

        outputs = []
        for q in range(len(inputs)):
            self.concat_inputs[q] = np.concatenate((self.hidden_states[q - 1], inputs[q]))

            self.forget_gates[q] = sigmoid(self.wf @ self.concat_inputs[q] + self.bf)
            self.input_gates[q] = sigmoid(self.wi @ self.concat_inputs[q] + self.bi)
            self.candidate_gates[q] = np.tanh(self.wc @ self.concat_inputs[q] + self.bc)
            self.output_gates[q] = sigmoid(self.wo @ self.concat_inputs[q] + self.bo)

            self.cell_states[q] = self.forget_gates[q] * \
                                  self.cell_states[q - 1] + self.input_gates[q] * self.candidate_gates[q]
            self.hidden_states[q] = self.output_gates[q] * np.tanh(self.cell_states[q])

            outputs += [self.wh @ self.hidden_states[q] + self.bh]

        return outputs

    def backward(self, errors, inputs):
        d_wf, d_bf = np.zeros_like(self.wf), np.zeros_like(self.bf)
        d_wi, d_bi = np.zeros_like(self.wi), np.zeros_like(self.bi)
        d_wc, d_bc = np.zeros_like(self.wc), np.zeros_like(self.bc)
        d_wo, d_bo = np.zeros_like(self.wo), np.zeros_like(self.bo)
        d_wh, d_bh = np.zeros_like(self.wh), np.zeros_like(self.bh)

        dh_next, dc_next = np.zeros_like(self.hidden_states[0]), np.zeros_like(self.cell_states[0])
        for q in reversed(range(len(inputs))):
            error = errors[q]

            d_wh += error @ self.hidden_states[q].T
            d_bh += error

            # Hidden State.
            d_hs = self.wh.T @ error + dh_next

            # Output Gate.
            d_o = np.tanh(self.cell_states[q]) * d_hs * sigmoid(self.output_gates[q], derivative=True)
            d_wo += d_o @ inputs[q].T
            d_bo += d_o

            # Cell State.
            d_cs = tanh(tanh(self.cell_states[q]), derivative=True) * self.output_gates[q] * d_hs + dc_next

            # Input Gate.
            d_i = d_cs * self.candidate_gates[q] * sigmoid(self.input_gates[q], derivative=True)
            d_wi += d_i @ inputs[q].T
            d_bi += d_i

            # Candidate Gate.
            d_c = d_cs * self.input_gates[q] * tanh(self.candidate_gates[q], derivative=True)
            d_wc += d_c @ inputs[q].T
            d_bc += d_c

            # Forget Gate.
            d_f = d_cs * self.cell_states[q - 1] * sigmoid(self.forget_gates[q], derivative=True)
            d_wf += d_f @ inputs[q].T
            d_bf += d_f

            # Concatenated.
            d_z = self.wf.T @ d_f + self.wi.T @ d_i + self.wc.T @ d_c + self.wo.T @ d_o

            dh_next = d_z[:self.hidden_size, :]
            dc_next = self.forget_gates[q] * d_cs

        for d in [d_wf, d_bf, d_wi, d_bi, d_wc, d_bc, d_wo, d_bo, d_wh, d_bh]:
            np.clip(d, -1, 1, out=d)

        self.wf += d_wf * self.learning_rate
        self.bf += d_bf * self.learning_rate

        self.wi += d_wi * self.learning_rate
        self.bi += d_bi * self.learning_rate

        self.wc += d_wc * self.learning_rate
        self.bc += d_bc * self.learning_rate

        self.wo += d_wo * self.learning_rate
        self.bo += d_bo * self.learning_rate

        self.wh += d_wh * self.learning_rate
        self.bh += d_bh * self.learning_rate

    def train(self, inputs, labels):
        inputs = [one_hot(input) for input in inputs]

        for _ in tqdm(range(self.num_epochs)):
            predictions = self.forward(inputs)

            errors = []
            for q in range(len(predictions)):
                errors += [-softmax(predictions[q])]
                errors[-1][char_to_idx[labels[q]]] += 1

            self.backward(errors, self.concat_inputs)

    def test(self, inputs, labels):
        accuracy = 0
        probs = self.forward([one_hot(input) for input in inputs])

        output = ''
        for q in range(len(labels)):
            prediction = idx_to_char[np.random.choice([*range(unique_chars)],
                                                      p=softmax(probs[q].reshape(-1)))]

            output += prediction

            if prediction == labels[q]:
                accuracy += 1

        print(f'True Data:\nt{labels}\n')
        print(f'Predictions:\nt{"".join(output)}\n')

        print(f'Accuracy: {accuracy * 100 / len(inputs)} %')


data = """
Nor marble nor the gilded monuments
Of princes shall outlive this powerful rhyme,
But you shall shine more bright in these contents
Than unswept stone besmeared with sluttish time.
When wasteful war shall statues overturn,
And broils root out the work of masonry,
Nor Mars his sword nor war’s quick fire shall burn
The living record of your memory.
’Gainst death and all-oblivious enmity
Shall you pace forth; your praise shall still find room
Even in the eyes of all posterity
That wear this world out to the ending doom.
So, till the Judgement that yourself arise,
You live in this, and dwell in lovers’ eyes.
""".lower()

chars = set(data)
data_size, unique_chars = len(data), len(chars)

print(f'Data size: {data_size}')

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

train_X, train_y = data[:-1], data[1:]

hidden_size = 25
lstm = LSTM(input_size=unique_chars + hidden_size,
            hidden_size=hidden_size, output_size=unique_chars,
            num_epochs=300, learning_rate=0.05)

lstm.train(train_X, train_y)
lstm.test(train_X, train_y)
