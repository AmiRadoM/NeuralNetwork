import random

class NN:
    def __init__(self, dimensions, initWeightRange = [-0.3, 0.3]):
        self.layers: list[Layer] = []
        for d in dimensions:
            self.layers.append(Layer(d))

        # Initiating the weights (TODO: obviously needs refactoring for efficiency)
        for i in range(len(self.layers)):
            if i+1 < len(self.layers):
                for n in self.layers[i].nodes:
                        for m in self.layers[i+1].nodes:
                            n.weights[m] = random.uniform(initWeightRange[0], initWeightRange[1])

    def predict(self, input):
        if(len(input) != len(self.layers[0].nodes)):
            raise Exception(f"Expected {len(self.layers[0].nodes)} inputs for neural network")
        
        for i in range(len(self.layers[0].nodes)):
            self.layers[0].nodes[i].value = input[i]
        
        # Going over every layer and calculating the value for every node based on the previous layer
        # TODO: also needs refactoring for efficiency
        for i in range(1, len(self.layers)):
            for n in self.layers[i].nodes:
                for m in self.layers[i-1].nodes:
                    n.value += m.value * m.weights[n]
        
        output = []
        for o in self.layers[len(self.layers) - 1].nodes:
            output.append(o.value)

        return output

class Layer:
    def __init__(self, nodes):
        self.nodes: list[Node] = []
        for i in range(nodes):
            self.nodes.append(Node())


class Node():
    def __init__(self):
        self.value = 0
        self.weights = {}