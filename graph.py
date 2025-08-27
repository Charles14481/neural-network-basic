import node
import model
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

class Graph:
    def __init__(self, width=5, layers=10, inNodes=1):
        self.width = width
        self.layers = layers
        self.inNodes = inNodes
    
    # Display NumPy array of nodes
    def display(self, nodes, property="bias"):
        # NumPy arrays for each layer to fill final graph
        G = np.empty(shape=(self.layers+2), dtype=object)
        
        for layer in range(self.layers+2):
            G[layer] = nx.Graph()
        
        G_main = nx.Graph()

        # Add nodes
        for arr in nodes:
            for layer in range(0,self.layers+2):
                n = arr[layer]
                if n != None:
                    # v = f"{n.__dict__[property]:.2f}"
                    v = f"{n.partials[0]:.10f}"
                    G[layer].add_node(n.pos, subset=n.layer, value=v)
                    G_main.add_node(n.pos, subset=n.layer, value=v)

        # Add edges
        for layer in range(1, self.layers+2):
            for n in G[layer]:
                if (n != None):
                    for n2 in G[layer-1]:
                        G_main.add_edge(n, n2, weight="%.2f" % round(nodes[n[0]][n[1]].weights[n2[0]], 2))

        # print(list(G_main.nodes(data=True)))

        # Show node graph
        pos = nx.multipartite_layout(G_main, subset_key="subset")
        values = nx.get_node_attributes(G_main, "value") 
        nx.draw(G_main, pos, labels=values)
        edges = nx.get_edge_attributes(G_main, "weight")
        nx.draw_networkx_edge_labels(G_main, pos, font_size=4, edge_labels=edges)
        plt.show()

if __name__=="__main__":
    width = 5
    layers = 10
    inNodes = 1
    g = Graph(width,layers,inNodes)
    m = model.Model(width=width,layers=layers, inputs=inNodes)
    g.display(m.generate_graph(width, layers, inNodes))