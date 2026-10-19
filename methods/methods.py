import numpy as np


def func(x: float, y: float) -> np.float64:
    return np.cos(x) - y

def exact_solution(x: float) -> np.float64:
    return (np.cos(x) + np.sin(x) -np.exp(-x)) / 2

def euler_method(x_0, y_0, h, n):
    x = np.linspace(x_0, x_0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y_0
    for i in range(n):
        y[i + 1] = y[i] + h * func(x[i], y[i])
    return x, y

def rk2_method(x_0, y_0, h, n):
    x = np.linspace(x_0, x_0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y_0
    for i in range(n):
        k_1 = func(x[i], y[i])
        k_2 = func(x[i] + h / 2, y[i] + h * k_1 / 2)
        y[i + 1] = y[i] + h * k_2
    return x, y

def rk4_method(x_0, y_0, h, n):
    x = np.linspace(x_0, x_0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y_0
    for i in range(n):
        k_1 = func(x[i], y[i])
        k_2 = func(x[i] + h / 2, y[i] + h * k_1 / 2)
        k_3 = func(x[i] + h / 2, y[i] + h * k_2 / 2)
        k_4 = func(x[i] + h, y[i] + h * k_3)
        y[i + 1] = y[i] + h * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    return x, y
