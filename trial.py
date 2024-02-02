import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Dummy data
input_size = 10
output_size = 2
hidden_size = 20
batch_size = 64

# Check if CUDA is available and move the model and data to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Instantiate the model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy input and target, move to device
input_data = torch.randn(batch_size, input_size).to(device)
target = torch.randint(0, output_size, (batch_size,)).to(device)

# Training loop
epochs = 100000
for epoch in range(epochs):
    # Forward pass
    output = model(input_data)
    
    # Compute the loss
    loss = criterion(output, target)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss for every epoch
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'simple_model.pth')
