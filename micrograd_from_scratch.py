
## import libraries 

import math 
import numpy as np
import matplotlib.pyplot as plt 

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

print(output._op)