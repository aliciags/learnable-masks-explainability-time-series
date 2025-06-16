# Load and shape synthetic data
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.simple import SimpleCNN


x = None
y = None

# load the data from synthetic data
data_folder = "./data/synthetic/train_5"
data_files = os.listdir(data_folder)
for file in data_files:
    if "samples_0" in file:
        if x is None and y is None:
            x = np.load(os.path.join(data_folder, file))
            y = np.zeros(5000)
        else:
            x = np.concatenate([x, np.load(os.path.join(data_folder, file))])
            y = np.concatenate([y, np.zeros(5000)])
    elif "samples_1" in file:
        if x is None and y is None:
            x = np.load(os.path.join(data_folder, file))
            y = np.ones(5000)
        else:
            x = np.concatenate([x, np.load(os.path.join(data_folder, file))])
            y = np.concatenate([y, np.ones(5000)])
    else:
        print("File not recognized")
        continue
    
x = x[:, np.newaxis, :]

print(x.shape)
print(y.shape)

# convert the data to torch tensors
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# create the dataset
dataset = TensorDataset(x, y)

# create the DataLoader
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# to CUDA if available otherwise to CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
print(f"Using device: {device}")


def train_model(model, train_loader, epochs=20, lr=0.001, device=device):
    model.to(device)  # Move model to GPU if available

    criterion = nn.CrossEntropyLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    # implement early stopping
    best_loss = float('inf')
    patience = 5
    counter = 0
    stopping_threshold = 0.001

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Early stopping
        if (best_loss - running_loss) > stopping_threshold:
            print(f"Loss improved from {best_loss:.4f} to {running_loss:.4f}")
            best_loss = running_loss
            counter = 0
        else:
            counter += 1
            print(f"Loss did not improve. Counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        train_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

    print("Training complete!")


# train the SimpleCNN model
model = SimpleCNN(in_channels=1, out_channels=2, hidden_size=64, kernel_size=5)

# Train the model
train_model(model, train_loader, epochs=500, lr=0.001, device=device)

torch.save(model.state_dict(), "./model/checkpoint/simpleCNN_5.pth")
