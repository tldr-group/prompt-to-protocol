import torch
import torch.nn as nn
import torch.nn.functional as F
import pybamm
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_ih = nn.Linear(3, 3, bias=True)   # 3×3 + 3 = 12
        self.W_hh = nn.Linear(3, 3, bias=False)  # 3×3     =  9
        self.out  = nn.Linear(3, 1, bias=True)   # 1×3 + 1 =  4
        # 12 + 9 + 4 = 25 total parameters

    def forward(self, x_t, h_prev):
        h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))
        y_t = torch.sigmoid(self.out(h_t)) * 9 + 3
        return y_t, h_t


def nn_in_pybamm(X, H_prev, Ws, bs):
    W_ih, W_hh, W_out = [w.tolist() for w in Ws]
    b_ih, b_out       = [b.tolist() for b in bs]

    # Linear transform + bias
    h_lin = []
    for i in range(len(W_ih)):
        z = pybamm.Scalar(b_ih[i])
        for j in range(len(X)):
            z += pybamm.Scalar(W_ih[i][j]) * X[j]
            z += pybamm.Scalar(W_hh[i][j]) * H_prev[j]
        h_lin.append(z)

    # Tanh activation
    h_t = [ (pybamm.exp(z) - pybamm.exp(-z)) / (pybamm.exp(z) + pybamm.exp(-z)) for z in h_lin ]

    # Output layer + sigmoid scaled
    lin_out = pybamm.Scalar(b_out[0])
    for j in range(len(h_t)):
        lin_out += pybamm.Scalar(W_out[0][j]) * h_t[j]
    y_sig = 1 / (1 + pybamm.exp(-lin_out))
    y_t   = -pybamm.Scalar(9) * y_sig -3
    return y_t, h_t

