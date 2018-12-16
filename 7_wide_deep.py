import numpy as np
import torch

data = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(data[:, 0:-1])
y_data = torch.from_numpy(data[:, -1])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out1 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out1))

        return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

