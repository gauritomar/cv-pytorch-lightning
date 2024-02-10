import torch
import torchvision
import torch.nn as nn # All neural network modules
import torch.optim as optim # All optimisers like SGD, Adam etc
import torch.nn.functional as F # Parameterless function like some Activation Functions
from torch.utils.data import DataLoader # Easier dataset management by creating mini batches
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For the MNIST dataset, the images are of dimensions: 28 x 28
# Hence the sequence length = 28 (we are treating each row of pixels as a time step)
# The input size = 28 ie. number of columns in each row of pixels as a time step

input_size = 28
sequence_length = 28
# sequence length refers to the number of input sequences that the model will process 

num_layers = 2
# the number of recurrent layers in the RNN architecture.
# Each layer consists of a sequence of recurrent units ie. LSTM or GRU cells 

hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Recurrent Neural Network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        self.input_size = input_size

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Prop LSTM
        out, _ = self.rnn(x, h0)
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