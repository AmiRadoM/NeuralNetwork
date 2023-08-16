from NN import NN
import activation
import random

# nn = NN([[1], [3, activation.sigmoid], [1]])
nn = NN([[1], [1]], lr=0.0001)

inputs = []
outputs = []

for i in range(100000):
    x = random.uniform(-1000,1000)
    inputs.append([x])
    outputs.append([x * 5])
nn.backpropagation(inputs,outputs)

print(nn.feedforward([10000]))