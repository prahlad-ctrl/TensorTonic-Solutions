import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x=np.array(x)
    t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t