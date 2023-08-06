import math

def linear(x):
    return x

def sigmoid(x):
    return (1 / (1 + math.pow(math.e,-x)))

def ReLU(x):
    return max(0, x)