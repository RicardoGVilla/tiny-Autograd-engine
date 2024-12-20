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
    ## Backpropagation and calculating gradiants for all intermediate nodes
    ## Adding a gradient attribute to the Value class for backpropagation
    ## Calculating backpropagation manually to better understand the logic behind the library 
    def __init__(self, data, _children=(), _op='', label=''): 
        self.data = data 
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = "e"
d = e + c; d.label = "d"
f = Value(-2.0, label='f')
L = d * f ; L.label = "L"
output = L


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
        if n._op:
            dot.node(name=uid+"_op", label=n._op, shape='ellipse', color='blue')
            # Create an oval node for the operation
            node_label = f"{n.label} | data: {n.data:.4f} | grad: {n.grad:.4f}" if n.label else f"data: {n.data:.4f}"
            dot.node(name=uid, label="{ %s }" % node_label, shape='record')
            dot.edge(uid+"_op", uid, style='dashed')
        else:
            # Update the leaf node format as well
            node_label = f"{n.label} | data: {n.data:.4f} | grad: {n.grad:.4f}" if n.label else f"data: {n.data:.4f}"
            dot.node(name=uid, label="{ %s }" % node_label, shape='record')
        
    # Add edges
    for n1, n2 in edges:
        if n2._op:
            # Edge from data to operation
            dot.edge(str(id(n1)), str(id(n2))+"_op")
        else:
            # Edge from operation to data (normally shouldn't happen without an op)
            dot.edge(str(id(n1))+"_op", str(id(n2)))
    
    # Save the graph to a file
    dot.render('computation_graph', format='png', cleanup=True)
    return dot



## L = d * f 
## dl/dd = ?f 

## rise over run 
## (f(x + h) - f(x))/h
## ((d+h)*f - (d)*f)/h
## (h*f)/h
## f

print("Output:", output)
dot = draw_dot(output)

L.grad = 1

def lol():
    h = 0.0001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label='f')
    L = d * f ; L.label = "L"
    output = L
    
    a = Value(2.0 + h, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label='f')
    L = d * f ; L.label = "L"
    output2 = L
    output2 = L.data
    
    
    ## calculating the rise over run 
    print((output/output2)/h)
    
    
    ## calculating the rise over run 
    print((output/output2)/h)

##lol()