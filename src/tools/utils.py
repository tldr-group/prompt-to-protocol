import inspect
import re
import os
import sys
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch.nn as nn
from src.tools.show_searching_space import show_searching_space


def extract_save_new_network_code(response, filename='nn_structure.py'):
    # Find all code blocks matching the pattern ```python...```
    code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
    
    # If any matches are found, extract the last code block
    if code_blocks:
        code_block = code_blocks[-1].strip()  # Get the last code block and strip any extra spaces
        
        # Save the last code block to the file
        with open(filename, 'w') as file:
            file.write(code_block)
        
        return code_block
    else:
        return None 

def extract_save_multi_network_code(response, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
    extracted_blocks = []
    for i, code_block in enumerate(code_blocks):
        code_block = code_block.strip()
        filename = os.path.join(output_dir, f"network_{i+1}.py")
        with open(filename, 'w') as file:
            file.write(code_block)
        extracted_blocks.append(code_block)
    return extracted_blocks

def save_model_info(module_name, result_folder, save_searching_space=True):
    nn_module = importlib.import_module(module_name)
    NeuralNetwork = nn_module.NeuralNetwork

    model_code_file = os.path.join(result_folder, 'model_architecture.py')
    imports = """   
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    model_source = inspect.getsource(NeuralNetwork)
    full_code = imports + "\n\n" + model_source

    with open(model_code_file, 'w') as f:
        f.write(full_code)
    
    if save_searching_space:
        # Show and save the searching space
        model_code_file = os.path.join(result_folder, 'searching_space.gif')
        # show_searching_space(nn_module, model_code_file)


def contains_rnn_keywords(network):
    for module in network.modules():
        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            return True
    return False

def preprocess_input(tensor, model):
    if contains_rnn_keywords(model):
        # Adjust the tensor shape to (batch_size, sequence_length, input_size)
        tensor = tensor.unsqueeze(0)  # Adding batch dimension (1, sequence_length, input_size)
    return tensor

def generate_search_space(model):
    space = {}
    for name, param in model.named_parameters():
        num_params = param.numel()
        for i in range(num_params):
            space[f'{name}_{i}'] = (-1.0, 1.0)
    return space
