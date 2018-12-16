import torch
import torch.nn as nn

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [[0, 1, 0, 2, 3, 3]] # 'hihell
one_hot_lookup = [[]]
cell = nn.LSTM(input_size=4, hidden_size=2, batch_first=True)

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

inputs = torch.tensor([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]]).float()
print('input size: ', inputs.size())
hidden = torch.randn(1, 3, 2), torch.randn(1, 3, 2)
out, hidden = cell(inputs, hidden)
print(out.size())