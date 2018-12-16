import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]])

model = Model()

criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print('\t', epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var = torch.tensor([[1.0]])
print('predict 1 hour: ', hour_var.item(), model(hour_var).item() > 0.5)
hour_var = torch.tensor([[7.0]])
print('predict 7 hour: ', hour_var.item(), model(hour_var).item() > 0.5)
