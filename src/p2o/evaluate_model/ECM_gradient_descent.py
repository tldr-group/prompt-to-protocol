import inspect
import os
import sys
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import numpy as np
from src.tools.show_searching_space import show_searching_space
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.tools.utils import *
from datetime import datetime
import multiprocessing as mp
import importlib
import os



# ECM parameters
R0_ref = 0.05  # reference resistance, unit: ohm
alpha = 0.01  # temperature coefficient of resistance, unit: 1/°C
T_ref = 25  # reference temperature, unit: °C
R_p_ref = 0.01  # polarization resistance, unit: ohm
C_p = 500  # polarization capacitance, unit: F
T_ambient = 25  # ambient temperature, unit: °C
h = 10  #  heat transfer coefficient, unit: W/(m^2·K)
A = 0.1  # surface area, unit: m^2
m = 0.2  # mass, unit: kg
c = 900  # specific heat capacity, unit: J/(kg·K)
dt = 100  # time step, unit: s 

heating_target = 5  # target heat generation, W

class ECMLayer(nn.Module):
    def __init__(self):
        super(ECMLayer, self).__init__()

    def forward(self, I, T):
        # Compute internal resistance
        R0 = R0_ref * (1 + alpha * (T - T_ref))
        R_p = R_p_ref * (1 + alpha * (T - T_ref))
        
        # Compute heat generation
        Q = I**2 * (R0 + R_p)
        
        # Update temperature dynamically
        T_next = T + dt * ((I**2 * (R0 + R_p) - h * A * (T - T_ambient)) / (m * c))
        
        return Q, R0, T_next
    
def penalize_large_models(net):
    param_count = sum(p.numel() for p in net.parameters())
    if param_count > 35:
        return 1e6, param_count  # Return a large loss if parameters exceed the limit
    return None, param_count
    
def evaluate_model(module_name, result_folder, random_seed=None):
    # Only set random seeds if explicitly provided, otherwise use true randomness
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        print(f"Starting evaluation for: {module_name} with random seed: {random_seed}")
    else:
        print(f"Starting evaluation for: {module_name} with true randomness (no fixed seed)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create an ECM layer instance
    ecm_layer = ECMLayer().to(device)
    
    # Import the neural network module and set up the device
    nn_module = importlib.import_module(module_name)
    NeuralNetwork = nn_module.NeuralNetwork
    net = NeuralNetwork().to(device)

    # Penalize large models
    penalty_loss, param_count = penalize_large_models(net)
    if penalty_loss is not None:
        best_loss_file = os.path.join(result_folder, "best_loss.txt")
        with open(best_loss_file, 'w') as f:
            f.write(str(penalty_loss))
        print(f"Model has too many parameters: {param_count}. Assigning large loss and exiting.")
        return result_folder, penalty_loss  # Exit immediately if the model is too large

    # Define the loss function
    def loss_function(predicted_heat, target_heat):
        return torch.mean((predicted_heat - target_heat) ** 2)

    def clip_gradients(model, max_norm):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    # def irregular_wave(t):
    #     return 5 * np.sin(0.0005 * t) + 3 * np.sin(0.003 * t)
    
    t_values = np.linspace(0, 3600, 36)  # Time range from 0 to 3600 seconds (1 hour)
    # irregular_current_values = torch.tensor([irregular_wave(t) for t in t_values], device=device, dtype=torch.float32)

    # Initialize temperature and polarization voltage
    T_initial = torch.tensor(25.0, device=device)
    T = T_initial.clone()
    target_heat_irregular = []
    # Ensure target temperature series aligns with t_values length for plotting
    # T_values_target = [T.item()]
    T_values_target = [T.item()] * (len(t_values) + 1)

    # for I in irregular_current_values:
    #     Q, R0, T = ecm_layer(I, T)
    #     target_heat_irregular.append(Q.item())
    #     T_values_target.append(T.item())
    
    # target_heat = torch.tensor(target_heat_irregular, dtype=torch.float32).to(device)
    target_heat = torch.full((len(t_values),), heating_target, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_values, dtype=torch.float32).to(device).unsqueeze(1)

    # Training the model
    num_epochs = 2000
    best_total_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()

        preprocessed_tensor = preprocess_input(t_tensor, net)
        preprocessed_tensor = preprocessed_tensor.to(device)

        # print(f"Preprocessed tensor shape: {preprocessed_tensor.shape}")

        current_output = net(preprocessed_tensor).flatten()
        T = T_initial.clone()
        T_values = [T]
        Q_values = []
        
        for I in current_output:
            Q, R0, T = ecm_layer(I, T)
            Q_values.append(Q)
            T_values.append(T)

        predicted_heat_tensor = torch.stack(Q_values)
        
        loss = loss_function(predicted_heat_tensor, target_heat)

        # Backward pass and optimization
        loss.backward()
        clip_gradients(net, max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        # Save best model
        if loss.item() < best_total_loss:
            best_total_loss = loss.item()
            best_model_state = net.state_dict()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

        best_loss_file = os.path.join(result_folder, "best_loss.txt")
        with open(best_loss_file, 'w') as f:
            f.write(str(best_total_loss))

    # Load the best model
    net.load_state_dict(best_model_state)

    # Evaluation
    net.eval()
    with torch.no_grad():
        current_output = net(t_tensor).flatten()
        T = T_initial.clone()
        predicted_heat = []
        T_values_pred = [T.item()]

        for I in current_output:
            Q, R0, T = ecm_layer(I, T)
            predicted_heat.append(Q)
            T_values_pred.append(T.item())
        
        predicted_heat_tensor = torch.stack(predicted_heat)

    # Plotting results
    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Plot Current
    axes[0].plot(t_values, current_output.cpu().numpy(), label='Predicted Current', color='#2E86AB', linewidth=2.5)
    axes[0].set_xlabel('Time [s]', fontsize=20)
    axes[0].set_ylabel('Current [A]', fontsize=20)
    axes[0].legend(fontsize=16, frameon=False)
    axes[0].tick_params(labelsize=16, width=1.5, length=6)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_linewidth(1.5)
    axes[0].spines['bottom'].set_linewidth(1.5)

    # Plot Predicted Heat vs Target Heat
    axes[1].plot(t_values, predicted_heat_tensor.cpu().numpy(), label='Predicted Heat', color='#A23B72', linewidth=2.5)
    axes[1].plot(t_values, target_heat.cpu().numpy(), label='Target Heat', linestyle='--', color='#F18F01', linewidth=2.5)
    axes[1].set_xlabel('Time [s]', fontsize=20)
    axes[1].set_ylabel('Heat Generation [W]', fontsize=20)
    axes[1].set_ylim(3, 8)  # Set y-axis range to 3-8
    axes[1].legend(fontsize=16, frameon=False)
    axes[1].tick_params(labelsize=16, width=1.5, length=6)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_linewidth(1.5)
    axes[1].spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    save_path = os.path.join(result_folder, 'results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return result_folder, best_total_loss


if __name__ == '__main__':

    # Multiprocessing tools

    def evaluate_model_wrapper(args):
        evaluate_model_func, module_name, result_folder = args
        best_total_loss = evaluate_model_func(module_name, result_folder) 
        return best_total_loss

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
                save_model_info(module_name, result_folder)
                args_list.append((evaluate_model_func, module_name, result_folder))
            
            for result_folder, best_total_loss in pool.imap(evaluate_model_wrapper, args_list):
                result_folders.append(result_folder)
                results.append(best_total_loss)
        
        return result_folders, results

    seed1_code = """
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 3)
        self.ln2 = nn.LayerNorm(3)  # subtle normalization change
        self.out = nn.Linear(3, 1)
        # Learnable gates
        self.alpha = nn.Parameter(torch.tensor(0.0))  # activation blend
        self.beta = nn.Parameter(torch.tensor(0.0))   # residual blend

    def forward(self, input):
        h1 = F.selu(self.fc1(input))  # early block from parents
        h2 = self.fc2(h1)
        h2 = self.ln2(h2)

        # Gated activation combining Seed 2's LeakyReLU and Seed 1's SELU
        a = torch.sigmoid(self.alpha)
        h2 = a * F.leaky_relu(h2, negative_slope=0.1) + (1.0 - a) * F.selu(h2)

        # Learnable residual mix between h1 and h2
        b = torch.sigmoid(self.beta)
        x = b * h2 + (1.0 - b) * h1

        x = self.out(x)
        output = torch.sigmoid(x) * 10  # Scale output to range [0, 10]
        return output

```
    """
    
    extract_save_new_network_code(seed1_code, 'seed1.py')


#   The available optimization methods are:
#   - `evaluate_model.ECM_gradient_descent`
#   - `evaluate_model.random_constant_heating`
#   - `evaluate_model.SAABO_constant_heating`

    result_folders, results = evaluate_models_in_parallel(
        module_names=['seed1'], 
        main_folder = 'experiments/Try', # The main folder to save the results
        evaluate_model_module='evaluate_model.ECM_gradient_descent', # The module containing the evaluate_model function
        num_workers=1
        )


    print("Result folders:", result_folders)
    print("Results:", results)