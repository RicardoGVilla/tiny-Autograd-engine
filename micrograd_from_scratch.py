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


## defining a class Value
class Value:
    ## Dunder methods (magic methods)
    ## Adding pointers to keep track of the children of the node
    ## Keep track of the operation that created the node
    ## Visualizing the graph
    ## Backpropagation and calculating gradiants for all intermediate nodes
    ## Adding a gradient attribute to the Value class for backpropagation
    ## Calculating backpropagation manually to better understand the logic behind the library 
    ## Calculating the gradient using the lol function
    ## Understanding neural networks and how they work 
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
    
    # Implement the tanh function (hyperbolic function)
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")
        return out


##Creating the first three nodes 
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')


## Creating the first operation (multiplication)
e = a * b; e.label = "e"

# Creating the second operation (addition)
d = e + c; d.label = "d"

# Creating the third operation (multiplication)
f = Value(-2.0, label='f')
L = d * f ; L.label = "L"

# output of the graph 
L

## building the graph with the trace function 
def trace(root):
    ## creating a set of nodes and edges 
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


# Calculating the gradients of L with respect to the nodes 

# gradient of L with respect to L
L.grad = 1

#gradient of L with respect to d
d.grad = f.data
#gradient of L with respect to f
f.grad = d.data


## setting up the variables 
# h = 0.0001
# d = 4000
# c = 1000
# e = -6000


#calculating the derivative of d with respect to L
# d = c + e

# result = d / c

# Calculating the slope run over
# ((c + h + e) - ((c + e)))/h
# which gives us:
# (c + h + e - c - e)/h
# Variable cancellation
# h/h

# Therefore we result with the following local slope (local derivative)
# d/c = 1

# Applying the chain rule
# L / c = L / d * d / c

## gradient of c with respect to L
c.grad = -2.000

## gradient of e with respect to L
e.grad = -2.000


# Calculating the derivative of L with respect to a 
# L / a = (L / e) * (e / a)

e = a * b
# e / a = 1

# L / a = (L / e) * (e / a)



## gradients of a with respect to L 
a.grad = -2.0 * -3.0

## gradients of b with respect to L 
b.grad = -2.0 * 2.0


# dot = draw_dot(L)

## defining a function to calculate the derivative of a function
# inline gradient calculation 
def lol():
    h = 0.0001


    ## Calculate the output of the function 
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label='f')
    L = d * f ; L.label = "L"
    output = L
    

    ## Calculate the output of the function with the input incremented by h
    a = Value(2.0 + h, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label='f')
    L = d * f ; L.label = "L"
    output = L.data
    output2 = L.data
    
    
    ## Calculating the derivative of the function with respect to the input
    print((output/output2)/h)

lol()   


# input values
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# The weights of the functions 
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# The bias of the function
b = Value(6.7, label="b")

# Calculating the output of the function
x1w1 = x1 * w1
x1w1.label = "x1*w1"

# Calculating the output of the function with the input increment by h 
x2w2 = x2 * w2
x2w2.label = "x2*w2"


# Calculating the output of the function with the input increment by h
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 +b; n.label = "n"

# Calculating the output of the function with the hyperbolic tangent function
o = n.tanh(); o.label = "o"

draw_dot(n)
