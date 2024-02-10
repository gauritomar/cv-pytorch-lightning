'''
LSTM is Long Short-Term Memory designed to address the issue of
vanishing gradients that commonly occur in traditional RNNs

LSTM is capable of learning and remembering long-term dependencies
in sequential data, making them suitable for tasks where understanding
context over long sequences in important.

LSTMs have a memory cell which allows them to learn when to remember or
forget information over time. The memory cell maintains a hidden state and
a cell state, which are updated over each time step of the sequence.

Cell State(C_t): is the memory of the LSTM unit. It is updated through the
process of forgetting or adding information using gates.

Forget Gate(F_t): controls what information from the previous cell state should
be forgotten or retained. It takes the previous hidden_state and current_inputs
and outputs a value between 0 and 1 for each element in cell state.

Input Gate(I_t): determine what new information should be added to the cell state
Also takes in the previous hidden state and current input as inputs and outputs a
value between 0 and 1 for each element in the cell state.

Output Gate(O_t): Controls what information should be output from the cell state.
Takes the previous hidden state and current inputs as inputs and outptus a value
between 0 and 1 for each element in the cell state.

Hidden State (H_t): Represents the output of the LSTM unit. It is computed based on
the current input, previous hidden state, and cell state.
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

input_size = 28
sequence_length = 28

num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # h0: initial hidden state
        # c0: initial cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # pass the hidden state and initial cell state as a tuple
        out, _ = self.lstm(x, (h0, c0))

        # the last hidden state contains information from all previous
        # states so we dont need to do conctentaion of all hidden states
        # just take the last hidden state

        # We are losing information by not contatenating all hidden states
        out = self.fc(out[:,-1,:])
        return out

train_dataset = datasets.MNIST(root='dataset/', train=True,
                        transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root='dataset/', train=True,
                        transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    for idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

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

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)