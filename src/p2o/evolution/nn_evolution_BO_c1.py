import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pandas as pd
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluate_model.ECM_gradient_descent import *  # Ensure this module is accessible

# Activation functions and their mappings
activation_functions = [
    'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU',
    'ELU', 'Softplus', 'Softsign', 'SiLU', 'GELU'
]

activation_dict = {
    'ReLU': nn.ReLU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU,
    'Softplus': nn.Softplus,
    'Softsign': nn.Softsign,
    'SiLU': nn.SiLU,
    'GELU': nn.GELU
}

normalization_options = [None, 'batchnorm', 'layernorm']

heating_target = 5  # target heat generation, W

# Custom neural network class
class CustomNet(nn.Module):
    def __init__(self, layers, activations, normalizations=None):
        super(CustomNet, self).__init__()
        self.layers = nn.ModuleList()

        input_dim = 1
        for i, (output_dim, activation_name) in enumerate(zip(layers, activations)):
            # Add Linear layer
            self.layers.append(nn.Linear(input_dim, output_dim))

            # Add normalization layer if specified
            if normalizations and i < len(normalizations):
                norm_type = normalizations[i]
                if norm_type == 'batchnorm':
                    self.layers.append(nn.BatchNorm1d(output_dim))
                elif norm_type == 'layernorm':
                    self.layers.append(nn.LayerNorm(output_dim))

            # Add activation function
            activation_class = activation_dict[activation_name]
            self.layers.append(activation_class())
            input_dim = output_dim

        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x) * 10  # Output scaling
        return x

# Generate random layers, activations, and normalizations
def generate_random_layers():
    num_layers = random.randint(0, 3)  #
    layers = [random.randint(1, 4) for _ in range(num_layers)]  # 
    activations = [random.choice(activation_functions) for _ in range(num_layers)]  # Random activations
    normalizations = []
    p_norm = 0.2  
    
    for _ in range(num_layers):
        if random.random() < p_norm:
            norm = random.choice(['batchnorm', 'layernorm'])
            normalizations.append(norm)
        else:
            normalizations.append(None)
    
    return layers, activations, normalizations

# Create and save initial models
def create_and_save_models(directory):
    os.makedirs(directory, exist_ok=True)
    loss_data = []

    for i in range(1, 11):
        layers, activations, normalizations = generate_random_layers()
        model = CustomNet(layers, activations, normalizations)
        file_path = os.path.join(directory, f"nn_{i}.pt")

        torch.save(model, file_path)
        
        loss_data.append([f"nn_{i}.pt", float('inf')])

        # Print the model architecture
        print(f"Model {i}:")
        print(model)
        print("-" * 40)
        
    # Save the loss data to a CSV file
    loss_df = pd.DataFrame(loss_data, columns=['File_path', 'Best_Total_Loss'])
    loss_df.to_csv(os.path.join(directory, "model_pool_loss_history.csv"), index=False)

# Evaluate the initial models
def load_and_evaluate_initial_models(directory, device):
    loss_data = []
    for i in range(1, 11):
        file_path = os.path.join(directory, f"nn_{i}.pt")
        model = torch.load(file_path).to(device)
        best_loss, param_count = train_and_evaluate_model(model, device)
        loss_data.append([f"nn_{i}.pt", best_loss])
        print(f"Initial Model {i} | Loss: {best_loss:.6f} | Params: {param_count}")

    # Save the loss data to a CSV file
    loss_df = pd.DataFrame(loss_data, columns=['File_path', 'Best_Total_Loss'])
    loss_df.to_csv(os.path.join(directory, "model_pool_loss_history.csv"), index=False)

# Penalize large models
def penalize_large_models(net):
    param_count = sum(p.numel() for p in net.parameters())
    if param_count > 35:
        print(f"Model has too many parameters: {param_count}. Assigning large loss.")
        return 1e6, param_count  # Return a large loss if parameters exceed the limit
    return None, param_count

# Training and evaluation function
def train_and_evaluate_model(net, device):
    penalty_loss, param_count = penalize_large_models(net)
    if penalty_loss is not None:
        return penalty_loss, param_count

    ecm_layer = ECMLayer().to(device)

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
    # irregular_current_values = torch.tensor(
    #     [irregular_wave(t) for t in t_values], 
    #     device=device, dtype=torch.float32
    # )

    # Initialize temperature and polarization voltage
    T_initial = torch.tensor(25.0, device=device)
    T = T_initial.clone()
    target_heat_irregular = []
    T_values_target = [T.item()]

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

    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        current_output = net(t_tensor).flatten()
        T = T_initial.clone()
        Q_values = []

        for I in current_output:
            Q, R0, T = ecm_layer(I, T)
            Q_values.append(Q)
            T_values_target.append(T.item())

        predicted_heat_tensor = torch.stack(Q_values)
        
        loss = loss_function(predicted_heat_tensor, target_heat)

        # Backward pass and optimization
        loss.backward()
        clip_gradients(net, max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        if loss.item() < best_total_loss:
            best_total_loss = loss.item()

    param_count = sum(p.numel() for p in net.parameters())
    return best_total_loss, param_count

# Objective function for Optuna
def objective(trial, device):
    # Suggest the number of layers
    num_layers = trial.suggest_int('num_layers', 0, 4)
    
    layers = []
    activations = []
    normalizations = []
    
    input_dim = 1
    
    for i in range(num_layers):
        # Suggest number of neurons
        n_units = trial.suggest_int(f'n_units_l{i}', 1, 5)
        layers.append(n_units)
        
        # Suggest activation function
        activation_name = trial.suggest_categorical(f'activation_l{i}', activation_functions)
        activations.append(activation_name)
        
        # Suggest normalization
        norm = trial.suggest_categorical(f'norm_l{i}', normalization_options)
        normalizations.append(norm)
    
    # Build the model
    model = CustomNet(layers, activations, normalizations)
    model.to(device)
    
    # Train and evaluate the model
    best_loss, param_count = train_and_evaluate_model(model, device)
    
    # Record the parameters and the loss
    trial.set_user_attr('layers', layers)
    trial.set_user_attr('activations', activations)
    trial.set_user_attr('normalizations', normalizations)
    trial.set_user_attr('param_count', param_count)
    
    return best_loss

# Main optimization function
def run_optimization(seed, num_iterations=30):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a unique base directory with timestamp and seed
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f"BO_{current_time}_seed_{seed}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create and save initial models
    create_and_save_models(base_dir)
    
    # Evaluate initial models
    load_and_evaluate_initial_models(base_dir, device)
    
    # Set up the Optuna study
    study = optuna.create_study(direction='minimize')
    
    total_iterations = num_iterations
    candidates_per_iteration = 3

    for iteration in range(total_iterations):
        print(f"Iteration {iteration+1}")
        trials = []
        log_data = []  # Initialize log data for this iteration
        
        for _ in range(candidates_per_iteration):
            trial = study.ask()
            try:
                # Directly call the objective function and log its result
                loss = objective(trial, device)
                study.tell(trial, loss)
                trials.append(trial)

                # Logging trial information
                layers = trial.user_attrs['layers']
                activations = trial.user_attrs['activations']
                normalizations = trial.user_attrs['normalizations']
                param_count = trial.user_attrs['param_count']
                log_data.append([iteration+1, trial.number, layers, activations, normalizations, param_count, loss])

            except Exception as e:
                print(f"Exception occurred: {e}")
        
        # Convert log_data to DataFrame and save the log
        log_df = pd.DataFrame(log_data, columns=[
            'Iteration', 'Trial', 'Layers', 'Activations', 'Normalizations', 'Param_Count', 'Loss'
        ])
        
        # Save the log data
        log_file = os.path.join(base_dir, "generation_log.csv")
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
        
        # Optionally, print the best trial so far
        best_trial = study.best_trial
        print(f"Best trial so far: Loss={best_trial.value}, Layers={best_trial.user_attrs['layers']}")
    
    print(f"Optimization completed for seed {seed}.")


if __name__ == "__main__":
    # Run optimization process for 10 different seeds
    # seed_list = [35, 28, 63, 74, 85, 96, 107, 118, 129]
    # seed_list = [76]
    seed_list = [100]

    for seed in seed_list:
        print(f"Running optimization with seed {seed}")
        run_optimization(seed, num_iterations=100)
