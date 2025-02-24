import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the Neural Network class with one hidden layer
class XOR_Network(nn.Module):
    def __init__(self):
        super(XOR_Network, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Input layer to hidden layer (2 inputs to 2 neurons)
        self.fc2 = nn.Linear(2, 1)  # Hidden layer to output layer (2 neurons to 1 output)
        self.relu = nn.ReLU()       # ReLU activation for hidden layer
        self.sigmoid = nn.Sigmoid() # Sigmoid activation for output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))   # Apply ReLU activation
        x = self.sigmoid(self.fc2(x))  # Apply Sigmoid activation
        return x

# Initialize the neural network
model = XOR_Network()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
num_epochs = 5000
loss_list = []

for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X)
    
    # Compute loss
    loss = criterion(outputs, y)
    
    # Backward pass (calculate gradients)
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Record the loss for plotting
    loss_list.append(loss.item())
    
    # Print loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the loss curve
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve for XOR Neural Network')
plt.grid(True)
plt.show()

# Test the model after training
with torch.no_grad():
    test_output = model(X)
    print("\nPredictions on XOR input:")
    print(test_output.round())  # Round the output to get binary predictions
