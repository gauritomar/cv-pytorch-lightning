'''
GRU combines the functionalities of the memory cell + hidden state in LSTM
into a single "gate" mechanism.

Update Gate(z): Controls how much of the past information should be passed
along to the future.

Reset Gate(z): Determines how much of the past information should be 
forgotten.

Current Memory Content(h_tilde): Candidate value for the new hidden state, 
computed based on the current input and the previous hidden state.

Hidden State(h_t): Output of the GRU unit and represents the current state
of the recurrent layer.
'''

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagation
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

train_dataset = datasets.MNIST(root='dataset/', train=True,
                            transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root='dataset/', train=False,
                            transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    
    for idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        if loader.dataset.train:
            print("Checking Accuracy on training data")
        else:
            print("Checking Accuracy on test data")

        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)

            _, predictions = scores.max(dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%")

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)