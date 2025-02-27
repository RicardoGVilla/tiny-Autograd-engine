## import libraries 
import math 
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from IPython.display import Image
import torch 
import random



    ## Dunder methods (magic methods)
    ## Adding pointers to keep track of the children of the node
    ## Keep track of the operation that created the node
    ## Visualizing the graph
    ## Backpropagation and calculating gradiants for all intermediate nodes
    ## Adding a gradient attribute to the Value class for backpropagation
    ## Calculating backpropagation manually to better understand the logic behind the library 
    ## Calculating the gradient using the lol function
    ## Understanding neural networks and how they work 
    ## Adding activation functions introduces non-linearity, which allows the network to model complex, non-linear relationships between inputs and outputs.Essential for tasks that require non linear decision boundaries such as image classification, speech recognition and natural language processing
    ## Coding the backpropagation process 
    ## Applying topological sort to the graph before backpropagation 
    ## Bug when using a variable more than one in the graph (accumulating gradients)
    ## Breaking down the tanh function into smaller parts
    ## Using pytorch to build a simple perceptron ( the simple type of neural network. A perceptron is a single layer neural network that can only solve linear problems) It cannot handle xor problems
    ## Building a simple neural network with a single neuron
    ## Building a layer or neurons
    ## Building a multi-layer perceptron (MLP)
    ## Adding parameters methods to the neuron and layer classes for easier access to the parameters
    ## Performing backward and forward passes on the MLP for training 
    ## Verifying the output after gradient descent






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
   
    def __init__(self, data, _children=(), _op='', label=''): 
        self.data = data 
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            ## the local derivative of the addition operation is 1
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad
        out._backward = _backward 

        return out
    
    ## substraction operation
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-1 * other)
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __neg__(self):
        return self * -1
    
    ## power operation
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad 
        out._backward = _backward
        return out
    
    ## division operation 
    def _truediv_(self,other):
        return self * other ** -1
    
    ## fall back to the multiplication operation if the other is not a Value
    def __rmul__(self, other):
        return self * other
    
    ## exponential operation 
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,),'expo')

        def _backward():
            self.grad = out.data + out.grad
        out._backward = _backward
        return out 
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            ## the local derivative of the multiplication operation is the other.data
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward

        return out
    
    # Implement the tanh function (hyperbolic function)
    ## the tanh function is a non linear function that maps any real number to the range [-1,1]
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            ## the local derivative of the tanh function is 1 - (self.data)**2
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward
        return out
    
    def backward(self):

        topo = []
        visited = set()
        ## calling _backward on each node in topological order
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.00

        for node in reversed(topo):
            node._backward() 
    
 

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
        # Create node label consistently for all nodes
        node_label = f"{n.label} | data: {n.data:.4f} | grad: {n.grad:.4f}" if n.label else f"data: {n.data:.4f} | grad: {n.grad:.4f}"
        
        if n._op:
            # Operation node
            dot.node(name=uid+"_op", label=n._op, shape='ellipse', color='blue')
            dot.node(name=uid, label="{ %s }" % node_label, shape='record')
            dot.edge(uid+"_op", uid, style='dashed')
        else:
            # Data node
            dot.node(name=uid, label="{ %s }" % node_label, shape='record')
        
    # Add edges
    for n1, n2 in edges:
        if n2._op:
            dot.edge(str(id(n1)), str(id(n2))+"_op")
        else:
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

# The bias of the neuron
b = Value(6.881373587019541, label="b")

# Calculating the output of the function
x1w1 = x1 * w1; x1w1.label = "x1*w1"

# Calculating the output of the function with the input increment by h 
x2w2 = x2 * w2; x2w2.label = "x2*w2"


# Calculating the output of the function with the input increment by h
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b; n.label = "n"

draw_dot(n)



o = n.tanh(); o.label = "o" 

## Calculating the derivatives 

# the derivative of o with respect to o (base case)
# o.grad = 1.0 

# calculating the derivative of o with respect to n 
# n.grad = o.grad * (1 - o.data**2)

# calculating the derivative of o with respect to x1w1x2ww2 
# x1w1x2w2.grad = 0.5 
# b.grad = 0.5 

# calculating the derivative of o with respect to x1w1 
# x1w1.grad = 0.5 
# x2w2.grad = 0.5 

# calculating the derivative of 0 with respect to x2, w2, x1, w1
# x2.grad = w2.data * x2w2.grad 
# w2.grad = x2.data * x2w2.grad 
# x1.grad = w1.data * x1w1.grad 
# w1.grad = x1.data * x1w1.grad 


# Obtaining the grad for all the nodes propagating backwards through the graph 
# o.grad = 1.00
# o.backward()
# n.backward()
# b.backward()
# x1w1x2w2.backward()
# x2w2.backward()
# x1w1.backward()

o.backward()

draw_dot(o)

x1 = torch.tensor([2.0]).double() ; x1.requires_grad = True
x2 = torch.tensor([0.0]).double() ; x2.requires_grad = True
w1 = torch.tensor([-3.0]).double() ; w1.requires_grad = True
w2 = torch.tensor([1.0]).double() ; w2.requires_grad = True
b = torch.tensor([6.881373587019541]).double() ; b.requires_grad = True

n = x1 * w1 + x2 * w2 + b 
o = torch.tanh(n)
print(o.data.item())
o.backward()

print("----")
print("x2", x2.grad.item())
print("w2", w2.grad.item())
print("x1", x1.grad.item())
print("w1", w1.grad.item())


 ## Building a simple neural network with a single neuron
class Neuron:
    ## nin is the number of inputs to the neuron
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    ## calling the neuron with the input x
    def __call__(self, x):
        # w * x + b 
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    ## getting the parameters of the neuron. returns the parameters scalar values
    def parameters(self):
        return self.w + [self.b]

# Building a layer or neurons
class Layer:
    ## nin is the numberof inputs to the layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    ## calling the layer with the input x 
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    ## getting the parameters of the layer. returns the parameters scalar values
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        

class MLP:
    ## nin is the number of inputs to the MLP
    ## nouts is the number of outputs to the MLP
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    ## getting the parameters of the MLP. returns the parameters scalar values
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

x = [2.0, 3.0, -1.0]
## Building a simple neural network that has 3 inputs, 2 hidden layers with 4 neurons each, and 1 output neuron
n = MLP(3, [4, 4, 1])
print(n(x))

draw_dot(n(x))

## Creating a simple dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

## Desired output
ys = [1.0, -1.0, -1.0, 1.0]


## Predicted output
ypred = [n(x) for x in xs]
print(ypred)


## Calculating the loss 
loss = Value(0.0)
for ygt, yout in zip(ys, ypred):
    loss = loss + (yout - ygt)**2
print("Loss function",loss)

loss.backward()



for p in n.parameters():
    p.data += -0.01 * p.grad

print("Loss after 1 step of gradient descent", n(x))
print("Ypred after 1 step of gradient descent", ypred)

# Training loop
for k in range(100):  # 100 epochs
    # Forward pass
    ypred = [n(x) for x in xs]
    
    # Compute loss
    loss = Value(0.0)
    for ygt, yout in zip(ys, ypred):
        loss = loss + (yout - ygt)**2
    
    # Backward pass
    for p in n.parameters():
        p.grad = 0.0  
    loss.backward()
    
    # Update weights
    learning_rate = 0.05  
    for p in n.parameters():
        p.data += -learning_rate * p.grad
    
    # Printing progress every 10 epochs
    if k % 10 == 0:
        print(f"Step {k}: loss = {loss.data}")

print("predictions after training", ypred)