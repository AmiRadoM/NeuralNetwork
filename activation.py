import math
import numpy

def linear(x):
    return x

def sigmoid(x):
    return (1 / (1 + numpy.power(math.e,-x)))

def ReLU(x):
    return max(0, x)