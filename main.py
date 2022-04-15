import matplotlib.pylab as plt
import numpy as np
#from common.functions import *
#from gradient import *
from Layer import*


layer_test = Relu_Layer()

x = np.array([[-1.0, 4.0], [2.0, -3.0]])

x_ = layer_test.forward(x)


print(x)
print(x_)

