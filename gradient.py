
import numpy as np


def numerical_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range (x.size):
        tmp = x[idx]
        x[idx] = float(tmp) + h
        f_delta_h2 = f(x)
        x[idx] = float(tmp) - h
        f_delta_h1 = f(x)
        grad[idx] = (f_delta_h2 - f_delta_h1) / (2 * h)
        x[idx] = tmp

    return grad


def numerical_gradient(f, x):
    if x.ndim == 1:
        return numerical_grad(f, x)
    else:
        grad = np.zeros_like(x)

        for idx, _ix in enumerate(x):
            grad[idx] = numerical_grad(f, _ix)

        return grad


def gradient_descent (f, start_pos, learning_rate = 0.01, iterations = 100):
    vec = start_pos
    vec_history = []

    for i in range(iterations):
        vec_history.append(vec.copy())
        #gradient = numerical_gradient(f, vec)
        #vec -= learning_rate * gradient
        grad = numerical_gradient(f, vec)
        diff = -learning_rate * grad
        vec += diff

    return vec, np.array(vec_history)


