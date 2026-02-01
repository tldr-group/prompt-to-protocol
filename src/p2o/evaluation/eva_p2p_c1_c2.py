import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from datetime import datetime
from src.tools.utils import *
import multiprocessing as mp
import importlib
import time
import random

# Multiprocessing tools

def evaluate_model_wrapper(args):
    evaluate_model_func, module_name, result_folder, random_seed = args
    best_total_loss = evaluate_model_func(module_name, result_folder, random_seed=random_seed) 
    return best_total_loss

def evaluate_models_in_parallel(module_names, main_folder, evaluate_model_module, num_workers=4, random_seeds=None):

    module_spec = importlib.import_module(evaluate_model_module)
    evaluate_model_func = getattr(module_spec, "evaluate_model") 

    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    
    # If random_seeds not provided, generate random seeds based on timestamp
    if random_seeds is None:
        # Use time-based random seeds to ensure different seeds for each model
        base_seed = int(time.time() * 1000000) % (2**31)  # Use microsecond timestamp
        random.seed(base_seed)
        random_seeds = [random.randint(0, 2**31 - 1) for _ in range(len(module_names))]
        print(f"Generated random seeds: {random_seeds}")
    elif len(random_seeds) != len(module_names):
        raise ValueError(f"Number of random_seeds ({len(random_seeds)}) must match number of module_names ({len(module_names)})")

    with mp.Pool(num_workers) as pool:
        result_folders = []
        results = []

        args_list = []
        for i, module_name in enumerate(module_names):
            result_folder = os.path.join(main_folder, f"{timestamp}_model_{i}")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            save_model_info(module_name, result_folder, save_searching_space=True)
            args_list.append((evaluate_model_func, module_name, result_folder, random_seeds[i]))
        
        for result_folder, best_total_loss in pool.imap(evaluate_model_wrapper, args_list):
            result_folders.append(result_folder)
            results.append(best_total_loss)
    
    return result_folders, results

if __name__ == '__main__':

    seed1_code = """
    ```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(3)
        self.fc2 = nn.Linear(3, 1, bias=False)
        
    def forward(self, input):
        x = self.fc1(input)
        x = self.bn1(x)  # Batch normalization
        x = torch.relu(x)  # Using ReLU activation function
        x = self.fc2(x)
        output = torch.sigmoid(x) * 5  # Scale output to range [0, 10]
        return output

```
    """

    # Save to src/p2o/evaluation/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_save_new_network_code(seed1_code, os.path.join(script_dir, 'seed1.py'))

    result_folders, results = evaluate_models_in_parallel(
        module_names=['src.p2o.evaluation.seed1'], 
        main_folder = 'experiments/BO_TEST', # The main folder to save the results
        evaluate_model_module='src.p2o.evaluate_model.SAABO_constant_heating', # The module containing the evaluate_model function
        num_workers=1,
        random_seeds=[110]  
        )


    print("Result folders:", result_folders)
    print("Results:", results)