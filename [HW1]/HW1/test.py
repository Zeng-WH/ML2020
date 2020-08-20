import numpy as np
yhat = np.load('yhat.npy')
yval = np.load('yval.npy')
print(yhat[0:20,:])
print(yval[0:20,:])