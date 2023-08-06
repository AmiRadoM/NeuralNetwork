import random
from typing import Callable, Optional
import activation

class NN:
    def __init__(self, layers: list[list[float, Optional[Callable]]], initWeightRange: tuple[float, float] = (-0.3, 0.3)):
        self.layers: list[Layer] = []
        for l in layers:
            # l[0] = number of nodes, l[1] = activation function
            if(len(l) < 2): l.append(activation.linear) # default null activation function to linear function
            self.layers.append(Layer(l[0], l[1]))

        # Initiating the weights (TODO: obviously needs refactoring for efficiency)
        for i in range(len(self.layers)):
            if i+1 < len(self.layers):
                for n in self.layers[i].nodes:
                        for m in self.layers[i+1].nodes:
                            n.weights[m] = random.uniform(initWeightRange[0], initWeightRange[1])

    def feedforward(self, input):
        if(len(input) != len(self.layers[0].nodes)):
            raise Exception(f"Expected {len(self.layers[0].nodes)} inputs for neural network")
        
        for i in range(len(self.layers[0].nodes)):
            self.layers[0].nodes[i].value = input[i]
        
        # Going over every layer and calculating the value for every node based on the previous layer
        # TODO: also needs refactoring for efficiency
        for i in range(1, len(self.layers)):
            for n in self.layers[i].nodes:
                for m in self.layers[i-1].nodes:
                    n.value += m.activation(m.value + m.bias) * m.weights[n]
        
        output = []
        for o in self.layers[len(self.layers) - 1].nodes:
            output.append(o.activation(o.value + o.bias))

        return output

    def backpropagation():
        pass

class Layer:
    def __init__(self, nodes, activation = activation.linear):
        self.nodes: list[Node] = []
        for i in range(nodes):
            self.nodes.append(Node(activation))


class Node():
    def __init__(self, activation = activation.linear):
        self.value = 0
        self.bias = 0
        self.activation = activation
        self.weights = {}