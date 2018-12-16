import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = Model()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

hour_var = torch.tensor([[4.0]])
print('predict (before training)', 4, model.forward(hour_var).item())

for epoch in range(500):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print('\t', epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print('predict (after training)', 4, model.forward(hour_var).item())
