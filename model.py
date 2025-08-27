# Creates MLP with forward and backward passes

import math
import sys
import warnings
import graph
from node import Node
from progressdisplay import ProgressDisplay
import csv
import numpy as np
import random
from datahandler import DataHandler as DH
from time import perf_counter

class Model:
    """Basic multi-layer perceptron for regression tasks"""
    def __init__(
            self,
            data,
            target,
            width=5,
            layers=1,
            normalization=None,
            batch_size=16,
            use_cat=True
        ) -> None:
        # Define instance variables
        self.width = width
        self.layers = layers
        """number of hidden layers"""
        self.batch_size = batch_size
        self.train_data, self.train_truths, self.test_data, self.test_truths = (None,) * 4
        self.losses = list()

        # Setup data
        self.train_data, self.train_truths, self.test_data, self.test_truths = DH.setup_data(data, target, normalization, use_cat)
        self.inputs = len(self.train_data[0])

        # Generate node graph structure
        self.nodes = self.generate_node_array(width, layers, self.inputs)
        
    def forward_pass(self, inputs, batch=0):
        """Generate prediction from inputs (a dict)"""
        # Set inputs
        for i, w in enumerate(inputs.values()):
            self.nodes[i][0].inputs[batch] = float(w)

        # print("fwd pass ", self.get_layer(self.nodes, 0))
        
        # Perform weighted sum calculations
        for lay in range(1, self.layers+2):
            for w in range(self.get_layer_size(lay)):
                n : Node = self.nodes[w][lay]
                n.w_sum(output=(lay==self.layers+1), batch=batch)

        return Node.activation(self.nodes[0][self.layers+1].inputs[batch], output=True)
    
    def calc_loss(self, preds, truths) -> np.float64:
        """Mean Squared Error (MSE) loss function."""
        assert len(preds) == len(truths)
        size = len(truths)
        mse = 0
        for i in range(size):
            mse += (truths[i] - preds[i]) ** 2
        return mse/size

    def backprop(self, error, batch):
        """Calculate partial derivatives for all nodes based on error (truth - pred)."""
        # print("\n", f" backprop with error {error:.2f}".center(40,'-'))

        #Compute error of output
        n_out = self.nodes[0][self.layers+1]
        n_out.partials[batch] = -2 * error * Node.activation(n_out.inputs[batch], output=True, derv=True)
        # print("partial of output:", n_out.partials[batch])
        
        # Backpropogate error, don't compute error for inputs
        for layer in range(self.layers, 0, -1):
            for row, n1 in enumerate(self.get_layer(self.nodes, layer)):
                # Error layer l+1 → n1
                delta = 0

                for n2 in self.get_layer(self.nodes, layer+1):
                    delta += n2.weights[row] * n2.partials[batch]

                newPartial = delta * Node.activation(n1.inputs[batch], output=False, derv=True)
                # assert newPartial != 0 # Model (probably) isn't perfect
                n1.partials[batch] = newPartial

                # if (random.randint(0,10) == 0):
                #     print(f"{n1.row}, {n1.layer}: partial is {newPartial}")
        
        # print('-' * 40)

    def grad_descent(self, batch_size, rate):
        """Update weights and biases based on partials calculated in backprop."""

        for layer in range(1, self.layers + 2): # Don't compute for inputs
            for n1 in self.get_layer(self.nodes, layer):
                # Clip gradient
                grad_norm = math.sqrt(sum(p**2 for p in n1.partials))
                if grad_norm > 1:
                    n1.partials = n1.partials * (1 / grad_norm)

                rob = rate/batch_size

                # Update weights ∀ connections between n1 and prev layer
                for row, n2 in enumerate(self.get_layer(self.nodes, layer-1)):
                    w_grad = 0
                    for batch in range(batch_size):
                        # w_grad += n1.partials[batch] * Node.activation(n2.inputs[batch], output=(layer==self.layers+1))
                        if n2.layer == 0:
                            prev_activation = n2.inputs[batch]
                        else:
                            prev_activation = Node.activation(n2.inputs[batch], output=False)
                        w_grad += n1.partials[batch] * prev_activation
                    
                    n1.weights[row] -= rob * w_grad
                    n1.weights[row] = np.clip(n1.weights[row], -1, 1)
                
                # Update bias for n
                n1.bias -= rob * sum(n1.partials)
                n1.bias = np.clip(n1.bias, -1, 1)

    def update_rate(self, type, info, epoch=None, c_rate=None):
        """use LR scheduler to generate new LR"""
        
        # Switching LR scheduler might take some train code modifications
        match type:
            case 'weird': # info will be (error, epsilon) <= (loss, constant)
                max_grad = 0
                for lay in range(0,self.layers+2):
                    for n in self.get_layer(self.nodes, lay):
                        max_grad = max(max_grad, np.mean(n.partials))
                
                max_grad /= self.batch_size
                return info[0] / (max_grad * np.sqrt(epoch)) * info[1]

            case 'step': # (factor, rate) <= (const, const)
                return c_rate * info[0] ** (epoch/info[1])
            
            case 'exp': # (rate) <= (const)
                return c_rate * (1-info[0]) ** epoch
            
            case _:
                raise ValueError("invalid update rate type")
                    
        # print(f"error: {error}, batch: {batch}, max_grad: {max_grad}, \tnewrate: {error / (max_grad * batch)}")

    def perform_batch(self, rate):
        """Perform SGD on mini batch."""
        # Setup training data
        # print(f"Performing {self.batch_size} batches".center(40, '-'))
        subset_i = random.sample(range(len(self.train_data)), self.batch_size)
        subset = np.empty(shape=self.batch_size, dtype=dict) # select input rows
        actuals = np.empty(shape=self.batch_size, dtype=np.float64) # corresponding actual values
        for i in range(self.batch_size):
            subset[i] = self.train_data[subset_i[i]]
            actuals[i] = self.train_truths[subset_i[i]]

        # print(Model.head(subset, length=1, maxc=100))

        # Forward and backward passes
        preds = np.empty(shape=self.batch_size, dtype=np.float64)

        for j in range(self.batch_size):
            # print(f"Batch {j}")
            prediction = self.forward_pass(subset[j], j)
            preds[j] = prediction
            self.backprop(actuals[j]-prediction, j)

        self.grad_descent(self.batch_size, rate)

        loss = self.calc_loss(preds, actuals)
        
        return loss

    def get_layer_size(self, lay):
        if lay == 0:
            return self.inputs
        if lay == self.layers+1:
            return 1
        return self.width
    
    def get_layer(self, nodes, layer):
        return np.fromiter(map(lambda r: nodes[r][layer], range(self.get_layer_size(layer))), dtype=object)
    
    def generate_node_array(self, width, layers, inputs):
        """Create a NumPy array of nodes; note that inputs and output are considered nodes and that array is not jagged."""
        nodes = np.empty(shape=(max(width, inputs), layers+2), dtype=object)

        for lay in range(0,layers+2):
            prev_nodes = None
            if lay != 0:
                prev_nodes = self.get_layer(nodes, lay-1)
            
            for r in range(self.get_layer_size(lay)):
                nodes[r][lay] = Node(row=r, layer=lay, prev=prev_nodes, batch_size=self.batch_size)
        print("generated nodes")

        return nodes

    def train(self, max_runs=None, target=0.5, learning_rate=0.05, update_params=('step', 0.5, 50), report_rate=None, use_rolling=True, display = None):
        """Repeatedly train model on batches until max_runs or converge within target is reached."""
        # Setup
        assert not (max_runs is None and target is None)
        if report_rate is None:
            report_rate = math.floor(max_runs/20) if max_runs is not None else -1
        stop = max_runs if max_runs is not None else int(1e+6)
        avg_loss = 0
        cache = np.full(50, np.inf)

        t_start = perf_counter()

        for r in range(1, stop+1):
            # Most work done here
            loss = self.perform_batch(learning_rate)

            # Post-process
            if update_params is not None:
                learning_rate = self.update_rate(update_params[0], update_params[1:], r, learning_rate)
            
            for lay in range(0,self.layers+2):
                for n in self.get_layer(self.nodes, lay):
                    n.partials.fill(0)

            avg_loss = (loss + (r-1) * avg_loss) / r
            cache = np.roll(cache, shift=-1)
            cache[-1] = loss
            rolling_loss = np.mean(cache)

            if (report_rate != -1 and r%report_rate==0):
                dt = (perf_counter() - t_start) * 1000
                t_start = perf_counter()
                if display is not None:
                    display.update(self.test(), dt)
                print(f"\tLoss for batch {r}: {loss:.4f}, took {dt:.3f} ms, cumulative is {avg_loss:.4f}, rolling is {sum(cache)/len(cache):.4f}")
                # print(cache)
            
            if (target is not None and (rolling_loss < target) if use_rolling else (avg_loss < target)):
                break
        
        return rolling_loss if use_rolling else avg_loss

    def test(self):
        """Determine model accuracy on dataset."""
        # I recommend not printing anything if you use progressdisplay
        # DH.head(self.test_data, name="test_data")
        # DH.head(self.test_truths, name="test_truths")

        preds = np.empty(shape=len(self.test_data), dtype=float)
        for i, data in enumerate(self.test_data):
            preds[i] = self.forward_pass(data)

        # DH.head(preds, name="predictions")
        
        return self.calc_loss(self.test_truths, preds)
    
    def show_graph(self, property="bias"):
        g = graph.Graph(self.width, self.layers, self.inputs)
        g.display(self.nodes, property)

if __name__=="__main__":
    
    m = Model(
        data='housing.csv',
        target='median_house_value',
        width=10,
        layers=3,
        batch_size=16,
        normalization="zscore",
        use_cat=True, # Cat implementation needs work, so turn off if necessary
    )
    
    # print('\t', m.nodes[1][7].bias)
    print(f"\nInitial test MSE: {m.test()}")
    print("avg loss was", 
        m.train(
            report_rate=10,
            learning_rate=0.1,
            max_runs=500,
        #   display=ProgressDisplay(interval=20),
        )
    )
    # print('\t', m.nodes[1][7].bias)
    print(f"Final test MSE:{ m.test()}")
    # dsp.keep()

    # print(m.get_layer(m.generate_node_array(m.width,m.layers,m.inputs), 0))
    #narr = m.generate_node_array(m.width,m.layers,m.inputs)
    # print (narr[0][2])
    # print(m.generate_node_array(m.width,m.layers,m.inputs))
    # m.display()

    # print("fpass: ", m.forward_pass(m.train_data[3])) # test algo on column, assumed to be convertible to int
    # print(m.perform_batch())
    # m.perform_batch()
    # m.show_graph(property="partials")