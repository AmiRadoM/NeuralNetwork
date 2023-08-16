import copy
import random
from typing import Callable, Optional
import activation

class NN:
    def __init__(self, layers: list[list[float, Optional[Callable]]], initWeightRange: tuple[float, float] = (-0.3, 0.3), lr  = 0.01):
        self.layers: list[Layer] = []        
        for l in layers:
            # l[0] = number of nodes, l[1] = activation function
            if(len(l) < 2): l.append(activation.linear) # default null activation function to linear function
            self.layers.append(Layer(l[0], l[1]))
        
        self.lr = lr

        # Initiating the weights (TODO: obviously needs refactoring for efficiency)
        for i in range(len(self.layers)):
            if i+1 < len(self.layers):
                for n in self.layers[i].nodes:
                        for m in range(len(self.layers[i+1].nodes)):
                            n.weights[m] = random.uniform(initWeightRange[0], initWeightRange[1])
    
    def reset_values(self):
        for i in range(1, len(self.layers)):
            for n in range(len(self.layers[i].nodes)):
                self.layers[i].nodes[n].value = 0

    def feedforward(self, input):
        if(len(input) != len(self.layers[0].nodes)):
            raise Exception(f"Expected {len(self.layers[0].nodes)} inputs for neural network")
        
        self.reset_values()
        
        for i in range(len(self.layers[0].nodes)):
            self.layers[0].nodes[i].value = input[i]
        
        # Going over every layer and calculating the value for every node based on the previous layer
        # TODO: also needs refactoring for efficiency
        for i in range(1, len(self.layers)):
            for n in range(len(self.layers[i].nodes)):
                for m in self.layers[i-1].nodes:
                    self.layers[i].nodes[n].value += m.activation(m.value + m.bias) * m.weights[n]
        
        output = []
        for o in self.layers[len(self.layers) - 1].nodes:
            output.append(o.value)

        return output

    def backpropagation(self, inputs, outputs):
        if(len(inputs) != len(outputs)):
            raise Exception("Number of inputs and outputs should be the same")
        for input in inputs:
            if(len(input) != len(self.layers[0].nodes)):
                raise Exception(f"Expected {len(self.layers[0].nodes)} inputs for neural network")
        for output in outputs:
            if(len(output) != len(self.layers[len(self.layers)-1].nodes)):
                raise Exception(f"Expected {len(self.layers[len(self.layers)-1].nodes)} outputs for neural network")
                    
        for iter in range(len(outputs)):
            o_error = []
            h_nodes = self.layers[len(self.layers)-1-1].nodes
            self.feedforward(inputs[iter])
            
            #Hidden to Output
            w_sums = []

            for o in range(len(outputs[iter])):
                #Getting outputs error
                #TODO: Try MSE (Mean Squared Error)
                o_error.append(outputs[iter][o] - self.layers[len(self.layers)-1].nodes[o].value)

                w_sum = 0
            
                for h in range(len(h_nodes)):
                    w_sum += h_nodes[h].weights[o]
                
                w_sums.append(w_sum)

                for h in range(len(h_nodes)):
                    print((o_error[o]))
                    h_nodes[h].weights[o] += (h_nodes[h].weights[o] / w_sum) * o_error[o] * self.lr
                    

            next_errors = []

            for h in range(len(h_nodes)):
                error = 0
                for w in range(len(h_nodes[h].weights)):
                    error += (h_nodes[h].weights[w] / w_sums[w]) * o_error[w]
                next_errors.append(error)
            
            #Hidden to Hidden + Input to Hidden
            for l in range(len(self.layers)-2).__reversed__():
                hidden1 = self.layers[l]
                hidden2 = self.layers[l+1]

                w_sums = []

                for h2 in range(len(hidden2.nodes)):
                    #Getting outputs error
                    #TODO: Try MSE (Mean Squared Error)

                    w_sum = 0
                
                    for h in range(len(hidden1.nodes)):
                        w_sum += hidden1.nodes[h].weights[h2]
                    
                    w_sums.append(w_sum)

                    for h in range(len(hidden1.nodes)):
                        hidden1.nodes[h].weights[h2] += (hidden1.nodes[h].weights[h2] / w_sum) * next_errors[h2] * self.lr

                prev_errors = copy.copy(next_errors)
                next_errors = []

                for h in range(len(hidden1.nodes)):
                    error = 0
                    for w in range(len(hidden1.nodes[h].weights)):
                        error += (hidden1.nodes[h].weights[w] / w_sums[w]) * prev_errors[w]
                    next_errors.append(error)
        

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