
import numpy as np


class Add_Layer:
    def __init__(self):
        pass

    def forward(self, x_1, x_2):
        out = x_1 + x_2

        return out

    def backward(self, dout):
        dx_1 = dout # dout * 1
        dx_2 = dout # dout * 1

        return dx_1, dx_2


class Mul_Layer:
    def __init__(self):
        self.x_1 = None
        self.x_2 = None

    def forward(self, x_1, x_2):
        self.x_1 = x_1
        self.x_2 = x_2
        out = self.x_1 * self.x_2

        return out

    def backward(self, dout):
        dx_1 = self.x_2 * dout
        dx_2 = self.x_1 * dout

        return dx_1, dx_2


class Sigmoid_Layer:
    def __init__(self):
        self.out = None

    def forward(self, x): # 1 / (1 + np.exp(-x))
        self.out = 1 / (1 + np.exp(-x))
        out = self.out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx


class Relu_Layer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
