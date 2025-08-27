import numpy as np
import random
import warnings

class Node():
    def __init__(
            self,
            layer=0,
            row=0,
            inputs=None,
            bias=None,
            batch_size=None,
            prev=None
        ) -> None:
        # Define instance variables
        self.layer = layer
        self.row = row
        self.pos = (row, layer)
        self.prev = prev
        self.bias = bias
        self.partials = None
        self.inputs = None
        
        if bias is None:
            # self.bias = np.random.uniform(-1,1)
            self.bias = np.random.uniform(-1,1)
        
        if batch_size is None:
            self.inputs = np.zeros(shape=100)
            self.partials = np.zeros(shape=100)
            print("no batch size")
        else:
            self.inputs = np.zeros(shape=batch_size)
            self.partials = np.zeros(shape=batch_size)

        self.weights = None

        # Setup weights as uniformly random floats
        if self.prev is not None:
            # self.weights = np.random.default_rng().uniform(-1, 1, size=len(prev))
            self.weights = np.random.default_rng().uniform(-1, 1, size=len(prev))
        
        elif layer != 0:
            print("Error 32")

    def w_sum(self, output, batch=0):
        """Set and return weighted sum of previous nodes"""
        _input = self.bias

        for i in range(len(self.prev)):
            if self.prev[i].layer == 0:
                prev_output = self.prev[i].inputs[batch]
            else:
                prev_output = Node.activation(self.prev[i].inputs[batch], output=False)
            _input += prev_output * self.weights[i]


        self.inputs[batch] = _input
        return _input
    
    def __repr__(self):
        """String info about node"""
        info = "Node info: "
        for k, v in vars(self).items():
            if k != "prev":
                info += f"{k}: {v}, "
        
        return info[:len(info)-2]

        #print("Node info: " + info[:len(info)-1])
        #return "%s(%r)" % (self.__class__, self.__dict__) # prints prev, which calls prev of previous nodes, etc. very long

    @staticmethod
    def activation(value, output=False, derv=False) -> int | np.float64:
        """Activation functions, derv=True: derivative of function"""

        # prevent operation overflow later through clipping
        value = np.clip(value, -1e12, 1e12)

        # Change strings below to change activation function type
        if output:
            type = "linear"
        else:
            type = "leaky_relu"
        
        match type:
            case "linear":
                value = np.clip(value, -500, 500)
                if (derv):
                    return 1
                return value
            
            case "relu":
                if (derv):
                    return 1 if value>0 else 0
                return max(0,value)
            
            case "leaky_relu":
                alpha = 0.01
                if (derv):
                    return 1 if value > 0 else alpha
                return value if value > 0 else alpha * value
            
            # Best for classification
            case "sigmoid":
                v2 = 1/(1+np.exp(-value))
                if (derv):
                    return v2 * (1-v2)
                return v2

            case _:
                raise ValueError("invalid activation type")