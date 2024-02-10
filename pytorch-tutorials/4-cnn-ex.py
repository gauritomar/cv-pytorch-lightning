import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import torchvision.transforms as transforms
'''
nout = [(nin + 2p - k)/s] + 1

nout: number of output features
nin: number of input features
k: convolution kernel size
p: convolution padding size
s: convolution stride size
'''

class CNN(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


model = CNN(in_channel=1, num_classes=10)
x = torch.rand(64, 1, 28, 28)
print(model(x).shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    for idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        data = data.to(device=device)
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
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")

        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            _, predictions = scores.max(dim =1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/(num_samples)*100:.2f}%")
    
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)