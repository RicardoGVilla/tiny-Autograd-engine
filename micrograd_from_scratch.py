## import libraries 
import math 
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from IPython.display import Image


## define function
def f(x):
    return 3*x**2 - 4*x + 5

## plot function
xs = np.arange(-5, 5, 0.25)
ys = f(xs)


## what is the derivative of f(x) = 3x^2 - 4x + 5 at any point x 
h = 0.001
x = 3.0 
(f(x + h) - f(x)) / h


# extra input values 
a = 2.0
b = - 3.0 
c = 10.0 
d = a*b + c 
# print(d)


# looking for the derivative of d with respect to a, b, and c 
a = 2.0
b = - 3.0 
c = 10.0 
d1 = a*b + c 
print(d)

d1 = a*b + c 
a += h 
d2 = a*b + c

# print("d1", d1)
# print("d2", d2)
# print("slope", (d2 - d1) / h)

## defining a class for a value

class Value:
    ## Dunder methods (magic methods)
    ## Adding pointers to keep track of the children of the node
    ## Keep track of the operation that created the node
    ## Visualizing the graph
    def __init__(self, data, _children=(), _op=''): 
        self.data = data 
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
output = a * b + c



def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    
    # Add nodes
    for n in nodes:
        uid = str(id(n))
        label = f"{{{n._op}|data: {n.data:.4f}}}" if n._op else f"data: {n.data:.4f}"
        dot.node(name=uid, label=label, shape='record')
        
    # Add edges
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))
    
    # Save the graph to a file
    dot.render('computation_graph', format='png', cleanup=True)
    return dot

print("Output:", output)
dot = draw_dot(output)

