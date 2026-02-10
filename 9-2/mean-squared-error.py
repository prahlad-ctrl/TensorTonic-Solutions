import numpy as np
import math

def mean_squared_error(y_pred, y_true):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mse = np.mean((y_pred - y_true)**2)
    return mse