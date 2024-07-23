from bench import Transformer, decode

import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Initialize the model
model = Transformer().to(device)

# Load the saved model parameters
model.load_state_dict(torch.load('model_params_2.pth'))

# Ensure the model is in evaluation mode
model.eval()

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(context, max_new_tokens=2000)

# Decode the generated indices to text
print(decode(generated_indices[0].tolist()))
