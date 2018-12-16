import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 32
train_loader = DataLoader(
    dataset=datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )),
    batch_size=batch_size,
    shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )),
    batch_size=batch_size,
    shuffle=True
)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 520)
        self.l2 = torch.nn.Linear(520, 320)
        self.l3 = torch.nn.Linear(320, 240)
        self.l4 = torch.nn.Linear(240, 120)
        self.l5 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))

        return self.l5(x)

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

def train(epoch):
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader, 0):

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train epoch: {} [{}/{} ({})]\tLoss: {}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))

def test():
    model.eval()

    test_loss = 0
    correct = 0

    for inputs, targets in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\tTest set: Average loss: {}, Accuracy: {}/{} ({})'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))


for epoch in range(5):
    train(epoch)
    test()