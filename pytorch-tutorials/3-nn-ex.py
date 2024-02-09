# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# create a fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 784 is the shape of the MNIST dataset
# 10 is the number of classes in the dataset
model = NN(784, 10)

# 64 values of size 784
x = torch.rand(64, 784)
print(x)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparamters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# initialise network

model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network

for epoch in range(num_epochs):

    for idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape) = torch.Size([64, 1, 28, 28])
        # because MNIST has just one channel, would be 3 in RGB
        # unroll the 28x28 matrix to a long vector

        data = data.reshape(data.shape[0], -1)

        # forward propagations
        scores = model(data)
        loss = criterion(scores, targets)

        # backward propagations
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        


# check accuracy on training and test to see how 
# good our model is

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # set model to evaludate mode
    model.eval()

    # we dont have to compute any gradients now
    with torch.no_grad():
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
        for x, y in loader:
            x = x.to(device= device)
            y = y.to(device= device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # the dimension of scores will be 64 x 10
            _, predictions = scores.max(dim=1) # take the max along dim=1
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%")

    model.train()
  

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
