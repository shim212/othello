# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, x.size)

    batch_size = x.shape[0]
    if batch_size == 1:
        x = x - np.max(x) # 오버플로 대책
        x = np.exp(x) / np.sum(np.exp(x))
    else:
        for i in range(batch_size):
            xi = x[i]
            xi = xi - np.max(xi) # 오버플로 대책
            xi = np.exp(xi) / np.sum(np.exp(xi))
            x[i] = xi
    return x


def mean_squared_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
    if t.ndim == 1:
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return 0.5 * np.sum((y-t)**2) / batch_size


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
    if t.ndim == 1:
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)
