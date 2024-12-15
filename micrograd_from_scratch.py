
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





    