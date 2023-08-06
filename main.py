from NN import NN
import activation

nn = NN([[1], [1, activation.sigmoid], [1, activation.sigmoid]])

print(nn.feedforward([1]))