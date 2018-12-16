import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# w = 1.0
w = torch.tensor([1.0], requires_grad=True)


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)

    return (y_pred - y) * (y_pred - y)


print('print (before training)', 4, forward(4).data[0])

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()

        print('\tgrad: ', x_val, y_val, w.grad.data[0], w.grad)

        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()

    print('progress: ', epoch, l.data[0])

print('print (after training', forward(4).data[0])
