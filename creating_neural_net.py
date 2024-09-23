import torch
import torch.nn as f
import numpy as np
from torchvision import transforms

# Define the activation functions
relu = f.ReLU()
sigmoid = f.Sigmoid()

# Input data (X1, X2, X3 from the table)
inputs = torch.tensor([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
    [1, -1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1]
], dtype=torch.float32)

# Weights from input layer to hidden layer
# These are the weights between X1, X2, X3 and X4, X5, X6
weights_input_hidden = torch.tensor([
    [3, -2, 1],   # Weights for X1 to X4, X5, X6
    [0, 3, 1],  # Weights for X2 to X4, X5, X6
    [1, 3, 2]    # Weights for X3 to X4, X5, X6
], dtype=torch.float32)

# Weights from hidden layer to output layer
# These are the weights between X4, X5, X6 and X7
weights_hidden_output = torch.tensor([-2, 2, 3], dtype=torch.float32)

# # Biases (assuming biases are zero here)
# bias_hidden = torch.tensor([0, 0, 0], dtype=torch.float32)
# bias_output = 0.0

# Forward propagation
for i, input_set in enumerate(inputs):
    # Compute the hidden layer activations
    hidden_input = torch.matmul(input_set, weights_input_hidden)
    hidden_output = relu(hidden_input)

    # Compute the output layer activation
    output_input = torch.matmul(hidden_output, weights_hidden_output)
    output = sigmoid(output_input)

    # Print the results
    print(f"Input {i+1}: {input_set.tolist()}, X4-X6: {hidden_output.tolist()}, X7: {output.item():.3f}")
