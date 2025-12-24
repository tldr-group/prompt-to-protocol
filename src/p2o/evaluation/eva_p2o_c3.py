
from datetime import datetime
from utils import *
import multiprocessing as mp
import importlib
import os

def evaluate_model_wrapper(args):
    evaluate_model_func, module_name, result_folder = args
    best_total_loss = evaluate_model_func(module_name, result_folder)
    return (result_folder, best_total_loss)

def evaluate_models_in_parallel(module_names, main_folder, evaluate_model_module, num_workers=4):

    module_spec = importlib.import_module(evaluate_model_module)
    evaluate_model_func = getattr(module_spec, "evaluate_model") 

    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    with mp.Pool(num_workers) as pool:
        result_folders = []
        results = []

        args_list = []
        for i, module_name in enumerate(module_names):
            result_folder = os.path.join(main_folder, f"{timestamp}_model_{i}")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            save_model_info_with_pybamm(module_name, result_folder)
            args_list.append((evaluate_model_func, module_name, result_folder))
        
        for result_folder, best_total_loss in pool.imap(evaluate_model_wrapper, args_list):
            result_folders.append(result_folder)
            results.append(best_total_loss)
    
    return result_folders, results

if __name__ == "__main__":
    # seed1_code = """
    #     ```python
    # import torch, numpy as np, pybamm
    # import torch.nn as nn
    # import pybamm

    # class NeuralNetwork(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.fc1 = nn.Linear(3, 3)
    #         self.ln1 = nn.LayerNorm(3, elementwise_affine=False)
    #         self.act = nn.LeakyReLU(0.01)
    #         self.fc2 = nn.Linear(3, 3)
    #         self.fc3 = nn.Linear(3, 1)

    #     def forward(self, x: torch.Tensor) -> torch.Tensor:
    #         h = self.fc1(x)
    #         h = self.ln1(h)
    #         h = self.act(h)
    #         h_res = self.fc2(h) + h
    #         h = self.act(h_res)
    #         out = torch.sigmoid(self.fc3(h)) * 5
    #         return out

    # def nn_in_pybamm(X, Ws, bs):
    #     W1, W2, W3 = [w.tolist() for w in Ws]
    #     b1, b2, b3 = [b.tolist() for b in bs]
    #     hidden = len(b1)
    #     assert hidden == 3
    #     # --- layer 1: affine ---
    #     lin1 = []
    #     for i in range(hidden):
    #         x = pybamm.Scalar(b1[i])
    #         for j in range(3):
    #             x += pybamm.Scalar(W1[i][j]) * X[j]
    #         lin1.append(x)

    #     # --- LayerNorm (no affine) ---
    #     mean1 = sum(lin1) / hidden
    #     var1 = sum((x - mean1) ** 2 for x in lin1) / hidden
    #     eps = 1e-5  # small constant to avoid division by zero
    #     inv_std1 = 1 / pybamm.sqrt(var1 + pybamm.Scalar(eps))
    #     ln1 = [(x - mean1) * inv_std1 for x in lin1]

    #     # --- activation ---
    #     h1 = [pybamm.maximum(x, 0.01 * x) for x in ln1]

    #     # --- layer 2: affine + residual + activation ---
    #     lin2 = []
    #     for i in range(hidden):
    #         x = pybamm.Scalar(b2[i])
    #         for j in range(hidden):
    #             x += pybamm.Scalar(W2[i][j]) * h1[j]
    #         lin2.append(x)
    #     h2 = [
    #         pybamm.maximum(lin2[i] + h1[i], 0.01 * (lin2[i] + h1[i]))
    #         for i in range(hidden)
    #     ]

    #     # --- output layer + sigmoid + scaling ---
    #     lin3 = pybamm.Scalar(b3[0])
    #     for j in range(hidden):
    #         lin3 += pybamm.Scalar(W3[0][j]) * h2[j]
    #     y = 1 / (1 + pybamm.exp(-lin3))
    #     I_scale = 5  # Scale output to current range
    #     return -pybamm.Scalar(I_scale) * y
    #     ```
    #     """

    seed1_code = """
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



    """

    seed2_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import pybamm
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_ih = nn.Linear(3, 2, bias=True)   # 3×2 + 2 =  8
        self.W_hh = nn.Linear(2, 2, bias=False)  # 2×2     =  4
        self.out  = nn.Linear(2, 1, bias=True)   # 1×2 + 1 =  3
        # 8 + 4 + 3 = 15 total parameters

    def forward(self, x_t, h_prev):
        h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))
        y_t = torch.sigmoid(self.out(h_t)) * 5 +4
        return y_t, h_t


def nn_in_pybamm(X, H_prev, Ws, bs):
    W_ih, W_hh, W_out = [w.tolist() for w in Ws]
    b_ih, b_out       = [b.tolist() for b in bs]

    h_lin = []
    for i in range(2):
        z = pybamm.Scalar(b_ih[i])
        for j in range(3):
            z += pybamm.Scalar(W_ih[i][j]) * X[j]
        for j in range(2):
            z += pybamm.Scalar(W_hh[i][j]) * H_prev[j]
        h_lin.append(z)

    h_t = [(pybamm.exp(z) - pybamm.exp(-z)) /
           (pybamm.exp(z) + pybamm.exp(-z)) for z in h_lin]

    lin_out = pybamm.Scalar(b_out[0])
    for j in range(2):
        lin_out += pybamm.Scalar(W_out[0][j]) * h_t[j]

    y_sig = 1 / (1 + pybamm.exp(-lin_out))
    y_t   = -pybamm.Scalar(5) * y_sig -4
    return y_t, h_t


    """

    seed3_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import pybamm
import numpy as np

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.W_ih = nn.Linear(3, 4, bias=False)   # 12 params
        self.W_hh = nn.Linear(4, 4, bias=False)   # 16 params
        self.out  = nn.Linear(4, 1, bias=False)   #  4 params

    def forward(self, x_t, h_prev):
        h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))
        y_t = torch.sigmoid(self.out(h_t)) * 5 +4
        return y_t, h_t


def nn_in_pybamm(X, H_prev, Ws):

    W_ih, W_hh, W_out = [w.tolist() for w in Ws]

    # ----- hidden update ----------------------------------------------------
    h_lin = []
    for i in range(4):
        z = pybamm.Scalar(0)                       # no bias
        for j in range(3):                         # input term
            z += pybamm.Scalar(W_ih[i][j]) * X[j]
        for j in range(4):                         # recurrent term
            z += pybamm.Scalar(W_hh[i][j]) * H_prev[j]
        h_lin.append(z)

    # tanh activation
    h_t = [(pybamm.exp(z) - pybamm.exp(-z)) /
           (pybamm.exp(z) + pybamm.exp(-z)) for z in h_lin]

    # ----- output layer ------------------------------------------------------
    lin_out = pybamm.Scalar(0)                     # no bias
    for j in range(4):
        lin_out += pybamm.Scalar(W_out[0][j]) * h_t[j]

    y_sig = 1 / (1 + pybamm.exp(-lin_out))
    y_t   = -pybamm.Scalar(5) * y_sig -4
    return y_t, h_t
    """


    # extract_save_new_network_code(seed1_code, 'ac_seed1.py')



    #   The available optimization methods are:
    #   - `evaluate_model.ECM_gradient_descent`
    #   - `evaluate_model.random_constant_heating`
    #   - `evaluate_model.SAABO_constant_heating`

    # result_folders, results = evaluate_models_in_parallel(
    #     module_names=['network_1', 'network_2', 'network_3','network_4','network_5', 'network_6', 'network_7', 'network_8', 'network_9', 'network_10'], # The list of module names to evaluate
    #     main_folder = 'Test_adaptive', # The main folder to save the results
    #     evaluate_model_module='evaluate_model.SAABO_adaptive', # The module containing the evaluate_model function
    #     num_workers=10
    #     )

    result_folders, results = evaluate_models_in_parallel(
        module_names=['ac_seed1'], # The list of module names to evaluate
        main_folder = 'Test_adaptive_RNN', # The main folder to save the results
        evaluate_model_module='evaluate_model.SAABO_adaptive_RNN', # The module containing the evaluate_model function
        num_workers=1
        )

