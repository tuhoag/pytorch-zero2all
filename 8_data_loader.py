import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    def __init__(self):
        data = np.loadtxt('./data/diabetes.csv',
                          delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(data[:, 0:-1])
        self.y_data = torch.from_numpy(data[:, -1])
        self.len = data.shape[0]


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


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


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        y_pred = model(inputs)

        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
