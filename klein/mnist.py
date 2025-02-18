import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Defines the model (3-layer neural network w/ ReLU activations between layers)
model = nn.Sequential(
    nn.Linear(28 * 28, 64), # Takes an input of size 28×28 = 784 pixels (flattened image) and maps it to 64 neurons.
    nn.ReLU(), # ReLU (Rectified Linear Unit) Activation Function is applied. It transforms negative values to 0 while keeping positive values the same.
    nn.Linear(64, 64), # Second hidden layer: Maps 64 neurons to another set of 64 neurons.
    nn.ReLU(),
    nn.Linear(64, 10) # Output layer: Maps 64 neurons to 10 output classes (digits 0-9).
)

#form chat, hooks to see activiations
activation = {}  # Dictionary to store activations

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu().numpy()  # Ensure activations are stored correctly
    return hook

# Register hooks for ReLU activations
model[1].register_forward_hook(get_activation('relu1'))
model[3].register_forward_hook(get_activation('relu2'))


# Defines the optimizer
params = model.parameters() # retrieves all trainable parameters (weights and biases) of the model
optimizer = optim.SGD(params, lr=1e-2) # Defines Stochastic Gradient Descent (SGD) as the optimizer; lr=1e-2 means the learning rate is 0.01


# Define loss
loss = nn.CrossEntropyLoss() #loss function, measures how far the model's predictions are from correct labels

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()) #loads mnist into data folder
train, val = random_split(train_data, [55000, 5000]) #splits 60k training samples into 55k for training, 5k for validation
train_loader = DataLoader(train, batch_size=32) #groups training data into batches of 32
val_loader = DataLoader(val, batch_size=32)

#from chat, uses CIFAR-10 to test out of distribution data
# Define transformation: Convert to grayscale and resize to 28x28
ood_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert RGB to Grayscale
    transforms.Resize((28, 28)),  # Resize to match MNIST
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize like MNIST
])

# Load CIFAR-10 test dataset as OOD data
ood_data = datasets.CIFAR10(root='data', train=False, download=True, transform=ood_transform)
ood_loader = DataLoader(ood_data, batch_size=1, shuffle=True)


# Training & Validation loop
nb_epochs = 5 # number of times the entire dataset will be passed through the model
for epoch in range(nb_epochs):
    losses = list()
    for batch in train_loader:
      x, y = batch # x: images (shape [32, 1, 28, 28]) , y: labels (digits 0-9)

      # Reshapes x from [32, 1, 28, 28] to [32, 784].
      batchSize = x.size(0)
      x = x.view(batchSize, -1) # -1 = 28 * 28 - Flatten images from [batch, 1, 28, 28] to [batch, 784].

      # Step 1: Forward
      logits = model(x) # passes x through the model, outputs logits

      # Step 2: Compute the objective function
      J = loss(logits, y) # computes cross-entropy loss between predictions and correct labels

      # Step 3: Cleaning the gradient
      model.zero_grad() # Clears previous gradients (prevents accumulation)
      # params.grad.zero_()

      # Step 4: Accumulate the partial derivatives of J wrt parameters
      J.backward() # Computes gradients of loss w.r.t. model parameters
      # params.grad.add_(dJ/dparams)

      # Step 5: Step in opposite direction of gradient
      optimizer.step() # Updates weights using SGD
      # with torch.no_grad(): params = params - eta * params.grad

      losses.append(J.item())


    print(f'Epoch {epoch+ 1}, training loss: {torch.tensor(losses).mean():.2f}') # Computes average loss and prints it.

    losses = list()
    for batch in val_loader:
      x, y = batch

      # x: b * 1 * 28 * 28
      batchSize = x.size(0)
      x = x.view(batchSize, -1) # -1 = 28 * 28

      # Step 1: Forward
      with torch.no_grad():
          logits = model(x)

      # Step 2: Compute the objective function
      J = loss(logits, y)

      losses.append(J.item())

    print(f'Epoch {epoch+ 1}, validation loss: {torch.tensor(losses).mean():.2f}')

correct = 0
total = 0
with torch.no_grad():
    for x, y in val_loader:
        batchSize = x.size(0)
        x = x.view(batchSize, -1)  # Flatten image
        logits = model(x)  # Forward pass
        predictions = torch.argmax(logits, dim=1)  # Get predicted class
        correct += (predictions == y).sum().item()  # Count correct predictions
        total += y.size(0)  # Count total samples

print(f"Validation Accuracy: {100 * correct / total:.2f}%")




# Select 10 images from the validation dataset
num_samples = 10
samples, labels = [], []

for i, (x, y) in enumerate(val_loader):
    if i == num_samples:
        break  # Stop after collecting 10 samples
    samples.append(x[0])  # Take first sample from the batch
    labels.append(y[0].item())  # Store label

# Convert to tensor and flatten images
samples = torch.stack(samples)  # Convert list to tensor
samples = samples.view(num_samples, -1)  # Flatten images

# Initialize dictionary to store activations
activation_values = {layer: [] for layer in activation.keys()}

# Pass each image through the model
for i in range(num_samples):
    model(samples[i])  # Forward pass

    # Store activations after each pass
    for layer in activation.keys():
        activation_values[layer].append(activation[layer])  # Store activations

# Convert activations to NumPy for easy plotting
for layer in activation_values:
    activation_values[layer] = np.array(activation_values[layer])

# Plot activations for different digits
for layer_name, act in activation_values.items():
    plt.figure(figsize=(12, 5))
    for i in range(num_samples):
        plt.plot(act[i], label=f"Digit {labels[i]}")  # Plot activation values

    plt.title(f"Activation Patterns in {layer_name}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Magnitude")
    plt.legend()
    plt.show()


# Get a single OOD (CIFAR-10) sample
ood_x, _ = next(iter(ood_loader))  # Get one CIFAR-10 sample
ood_x = ood_x.view(-1, 28 * 28)  # Flatten to match MNIST format

# Pass it through the model
model(ood_x)

# Visualize activations for OOD sample
for layer_name, act in activation.items():
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(act[0])), act[0])  # Remove `.cpu()`, since act[0] is already a NumPy array
    plt.title(f"OOD Activation in {layer_name} (CIFAR-10)")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Magnitude")
    plt.show()

import numpy as np

# Get a single MNIST image
mnist_x, _ = next(iter(val_loader))  # Get one MNIST test sample
mnist_x = mnist_x[0].view(-1, 28 * 28)  # Flatten

# Get a single OOD (CIFAR-10) image
ood_x, _ = next(iter(ood_loader))
ood_x = ood_x.view(-1, 28 * 28)  # Flatten

# Forward pass for both MNIST and OOD samples
model(mnist_x)
mnist_activations = {layer: activation[layer][0] for layer in activation}  # Remove `.cpu().numpy()`

model(ood_x)
ood_activations = {layer: activation[layer][0] for layer in activation}  # Remove `.cpu().numpy()`


# Plot comparisons
for layer_name in activation.keys():
    plt.figure(figsize=(10, 5))
    plt.plot(mnist_activations[layer_name], label="MNIST (In-Distribution)", color='blue')
    plt.plot(ood_activations[layer_name], label="OOD (CIFAR-10)", color='red')
    plt.title(f"Activation Comparison in {layer_name}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Magnitude")
    plt.legend()
    plt.show()
