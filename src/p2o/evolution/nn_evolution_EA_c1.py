import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from deap import base, creator, tools
from datetime import datetime
from termcolor import colored
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.p2o.evaluate_model.ECM_gradient_descent import *

# Activation functions
activation_functions = [
    nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU,
    nn.ELU, nn.Softplus, nn.Softsign, nn.SiLU, nn.GELU
]

heating_target = 5  # target heat generation, W

def penalize_large_models(net):
    param_count = sum(p.numel() for p in net.parameters())
    if param_count > 35:
        print(f"Model has too many parameters: {param_count}. Assigning large loss.")
        return 1e6, param_count  # Return a large loss if parameters exceed the limit
    return None, param_count

# Function to extract architecture from the model
def extract_architecture(model):
    layers_list = []
    activations_list = []
    normalization_list = []
    
    i = 0
    while i < len(model.layers):
        layer = model.layers[i]
        if isinstance(layer, nn.Linear):
            # Record the output features of the Linear layer
            layers_list.append(layer.out_features)
            
            # Check for normalization layer
            norm = None
            if i + 1 < len(model.layers):
                next_layer = model.layers[i + 1]
                if isinstance(next_layer, nn.BatchNorm1d):
                    norm = 'batchnorm'
                    i += 1
                elif isinstance(next_layer, nn.LayerNorm):
                    norm = 'layernorm'
                    i += 1
            normalization_list.append(norm)
            
            # Record the activation function
            if i + 1 < len(model.layers):
                activation_layer = model.layers[i + 1]
                activations_list.append(type(activation_layer))
                i += 1
        else:
            i += 1
    return layers_list, activations_list, normalization_list

class CustomNet(nn.Module):
    def __init__(self, layers, activations, normalizations=None):
        super(CustomNet, self).__init__()
        self.layers = nn.ModuleList()

        input_dim = 1
        for i, (output_dim, activation) in enumerate(zip(layers, activations)):
            # Add Linear layer
            self.layers.append(nn.Linear(input_dim, output_dim))
            
            # Add normalization layer with specified type if exists
            if normalizations and i < len(normalizations):
                norm_type = normalizations[i]
                if norm_type == 'batchnorm':
                    self.layers.append(nn.BatchNorm1d(output_dim))
                elif norm_type == 'layernorm':
                    self.layers.append(nn.LayerNorm(output_dim))
            
            # Add activation function
            self.layers.append(activation())
            input_dim = output_dim

        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x) * 10  # Output scaling
        return x

# Random generation of layers, activations, and normalizations
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

# Define custom mutation function
def custom_mutate(individual, indpb=0.1):
    print("Entering custom_mutate...")
    model = individual[0]
    mutated = False

    # Initialize a local random generator
    local_random = random.Random()
    local_random.seed(random.randint(1, 10000))  

    # Extract architecture components
    layers_list, activations_list, normalization_list = extract_architecture(model)

    # Log original architecture
    print("Original layers:", layers_list)
    print("Original activations:", [act.__name__ for act in activations_list])
    print("Original normalizations:", normalization_list)

    # Mutate the number of neurons in each layer
    for i in range(len(layers_list)):
        if local_random.random() < indpb:
            old_out_features = layers_list[i]
            new_out_features = max(1, min(10, old_out_features + local_random.choice([-1, 1])))
            if new_out_features != old_out_features:
                mutated = True
                layers_list[i] = new_out_features
                print(f"Mutated layer {i}: {old_out_features} -> {new_out_features}")

    # Mutate activation functions
    for i in range(len(activations_list)):
        if local_random.random() < (indpb):
            old_activation = activations_list[i]
            new_activation = local_random.choice(activation_functions)
            if old_activation != new_activation:
                mutated = True
                activations_list[i] = new_activation
                print(f"Mutated activation {i}: {old_activation.__name__} -> {new_activation.__name__}")

    # Mutate normalization layers
    for i in range(len(normalization_list)):
        if local_random.random() < (indpb):
            old_norm = normalization_list[i]
            action = local_random.choice(['add', 'remove', 'change'])
            if action == 'add' and old_norm is None:
                new_norm = local_random.choice(['batchnorm', 'layernorm'])
                normalization_list[i] = new_norm
                mutated = True
                print(f"Added normalization {i}: {new_norm}")
            elif action == 'remove' and old_norm is not None:
                normalization_list[i] = None
                mutated = True
                print(f"Removed normalization {i}")
            elif action == 'change' and old_norm is not None:
                new_norm = local_random.choice(['batchnorm', 'layernorm'])
                if new_norm != old_norm:
                    normalization_list[i] = new_norm
                    mutated = True
                    print(f"Changed normalization {i}: {old_norm} -> {new_norm}")

    if mutated:
        # Reconstruct the mutated model
        new_model = CustomNet(layers_list, activations_list, normalization_list)
        individual[0] = new_model
        print("Model mutated and reconstructed with new layers, activations, and normalizations.")
        print("New layers:", layers_list)
        print("New activations:", [act.__name__ for act in activations_list])
        print("New normalizations:", normalization_list)
    else:
        print("No mutation occurred for this individual.")

    return individual,

# Clone the model
def clone_model(model):
    layers_list, activations_list, normalization_list = extract_architecture(model)
    cloned_layers = layers_list.copy()
    cloned_activations = activations_list.copy()
    cloned_normalizations = normalization_list.copy()
    return CustomNet(cloned_layers, cloned_activations, cloned_normalizations)

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

# Load individuals from files
def load_individuals_from_files(directory, file_name="model_pool_loss_history.csv"):
    individuals = []
    loss_df = pd.read_csv(os.path.join(directory, file_name))
    
    for i in range(1, 11):
        file_path = os.path.join(directory, f"nn_{i}.pt")
        model = torch.load(file_path)  
        loss = loss_df[loss_df['File_path'] == f"nn_{i}.pt"]['Best_Total_Loss'].values[0]
        individual = creator.Individual([model])
        individual.fitness.values = (loss,)
        individuals.append(individual)
    
    return individuals

# Save parent model evaluation
def save_parent_evaluation(pool, directory):
    eval_data = []
    for i, individual in enumerate(pool):
        eval_data.append([f"nn_{i+1}.pt", individual.fitness.values[0]])

    eval_df = pd.DataFrame(eval_data, columns=['File_path', 'Best_Total_Loss'])
    eval_df.to_csv(os.path.join(directory, "parent_model_evaluation.csv"), index=False)

# Load parent evaluation
def load_parent_evaluation(directory):
    eval_file = os.path.join(directory, "parent_model_evaluation.csv")
    if os.path.exists(eval_file):
        eval_df = pd.read_csv(eval_file)
        return eval_df
    else:
        return None

# Evaluate parent models
def evaluate_parent_models(pool, device, directory):
    eval_df = load_parent_evaluation(directory)
    if eval_df is not None:
        for i, individual in enumerate(pool):
            individual.fitness.values = (
                eval_df[eval_df['File_path'] == f"nn_{i+1}.pt"]['Best_Total_Loss'].values[0],
            )
    else:
        for i, individual in enumerate(pool):
            model = individual[0].to(device)
            best_loss, param_count = train_and_evaluate_model(model, device)
            individual.fitness.values = (best_loss,)
            print(f"Parent Model {i+1} | Loss: {best_loss:.6f} | Params: {param_count}")
        save_parent_evaluation(pool, directory)

# Load a PyTorch model from a file
def load_pytorch_model(file_path):
    model = torch.load(file_path)
    print(f"Model loaded from {file_path}")
    print(model)  # Print the model architecture

# Create the DEAP creator and toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("select", tools.selTournament, tournsize=2)  
toolbox.register("mutate", custom_mutate, indpb=0.1)

# Generate offspring with crossover handling normalizations
def generate_offspring(pool, num_offspring):
    offspring = []
    for _ in range(num_offspring):
        parent1, parent2 = toolbox.select(pool, 2)
        
        # Print parent models
        print(colored(f'Parent1 Model: {parent1[0]}', 'yellow'))
        print(colored(f'Parent2 Model: {parent2[0]}', 'yellow'))
        
        # Extract architectures
        parent1_layers, parent1_activations, parent1_norms = extract_architecture(parent1[0])
        parent2_layers, parent2_activations, parent2_norms = extract_architecture(parent2[0])

        # Perform crossover
        if len(parent1_layers) > 1 and len(parent2_layers) > 1:
            crossover_point = random.randint(1, min(len(parent1_layers), len(parent2_layers)) - 1)
            child1_layers = parent1_layers[:crossover_point] + parent2_layers[crossover_point:]
            child2_layers = parent2_layers[:crossover_point] + parent1_layers[crossover_point:]

            child1_activations = parent1_activations[:crossover_point] + parent2_activations[crossover_point:]
            child2_activations = parent2_activations[:crossover_point] + parent1_activations[crossover_point:]

            child1_norms = parent1_norms[:crossover_point] + parent2_norms[crossover_point:]
            child2_norms = parent2_norms[:crossover_point] + parent1_norms[crossover_point:]
        else:
            # If either parent has only one layer, perform mutation instead of crossover
            child1_layers, child2_layers = parent1_layers.copy(), parent2_layers.copy()
            child1_activations, child2_activations = parent1_activations.copy(), parent2_activations.copy()
            child1_norms, child2_norms = parent1_norms.copy(), parent2_norms.copy()

            # Mutate the number of neurons in hidden layers
            if len(child1_layers) == 1:
                child1_layers[0] += random.randint(-1, 1)
                child1_layers[0] = max(1, child1_layers[0])  
            if len(child2_layers) == 1:
                child2_layers[0] += random.randint(-1, 1)
                child2_layers[0] = max(1, child2_layers[0])  

        # Create child models
        child1 = CustomNet(child1_layers, child1_activations, child1_norms)
        child2 = CustomNet(child2_layers, child2_activations, child2_norms)

        # Create DEAP individuals
        offspring.extend([creator.Individual([child1]), creator.Individual([child2])])

    # Apply mutation to each offspring
    for child in offspring:
        toolbox.mutate(child)  # Apply mutation

    return offspring[:num_offspring]

# Save offspring models
def save_offspring_models(offspring, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for i, child in enumerate(offspring):
        file_path = os.path.join(folder_name, f"nn_new_{i+1}.pt")
        torch.save(child[0], file_path)  
    return folder_name

# Update the pool with the best loss
def update_pool_best_loss(pool, directory, file_name="model_pool_loss_history.csv"):
    # Read existing loss data
    loss_file_path = os.path.join(directory, file_name)
    if os.path.exists(loss_file_path):
        loss_df = pd.read_csv(loss_file_path)
    else:
        # Create a new DataFrame if the file does not exist
        loss_df = pd.DataFrame(columns=['File_path', 'Best_Total_Loss'])
    
    new_data = []
    for i, ind in enumerate(pool):
        model_file_name = f"nn_{i+1}.pt"
        model_file_path = os.path.join(directory, model_file_name)
        print(f"Recording loss for model {model_file_name}: {ind.fitness.values[0]}")
        # torch.save(ind[0], model_file_path)  # Save the model to a .pt file
        
        # print(f"Saving model {model_file_name}:")
        # print(ind[0])
        
        new_data.append([model_file_name, ind.fitness.values[0]])
    
    new_df = pd.DataFrame(new_data, columns=['File_path', 'Best_Total_Loss'])
    combined_df = pd.concat([loss_df, new_df]).drop_duplicates(subset=['File_path'], keep='last')
    
    # Sort the DataFrame by loss and keep only the top 10 models
    combined_df = combined_df.sort_values(by='Best_Total_Loss').head(10)
    combined_df.to_csv(loss_file_path, index=False)

# Log generation data
def log_generation_data(generation, pool, offspring, directory):
    log_data = []
    for i, ind in enumerate(pool):
        log_data.append([
            generation, 
            f"Parent_nn_{i+1}.pt", 
            ind.fitness.values[0], 
            sum(p.numel() for p in ind[0].parameters())
        ])
    for i, child in enumerate(offspring):
        log_data.append([
            generation, 
            f"Child_nn_new_{i+1}.pt", 
            child.fitness.values[0], 
            sum(p.numel() for p in child[0].parameters())
        ])
    
    log_df = pd.DataFrame(log_data, columns=['Generation', 'Model', 'Loss', 'Parameters'])
    log_file = os.path.join(directory, "generation_log.csv")
    
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

# Train and evaluate the model
def train_and_evaluate_model(net, device):
    # Create an ECM layer instance

    penalty_loss, param_count = penalize_large_models(net)
    if penalty_loss is not None:
        print(f"Model has too many parameters: {param_count}. Assigning large loss.")
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

    def irregular_wave(t):
        return 5 * np.sin(0.0005 * t) + 3 * np.sin(0.003 * t)
    
    t_values = np.linspace(0, 3600, 36)  # Time range from 0 to 3600 seconds (1 hour)
    irregular_current_values = torch.tensor(
        [irregular_wave(t) for t in t_values], 
        device=device, dtype=torch.float32
    )

    # Initialize temperature and polarization voltage
    T_initial = torch.tensor(25.0, device=device)
    T = T_initial.clone()
    target_heat_irregular = []
    T_values_target = [T.item()]

    for I in irregular_current_values:
        Q, R0, T = ecm_layer(I, T)
        target_heat_irregular.append(Q.item())
        T_values_target.append(T.item())
    
    target_heat = torch.tensor(target_heat_irregular, dtype=torch.float32).to(device)
    # target_heat = torch.full((len(t_values),), heating_target, dtype=torch.float32, device=device)

    t_tensor = torch.tensor(t_values, dtype=torch.float32).to(device).unsqueeze(1)

    # Training the model
    num_epochs = 2000
    best_total_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        # current_output = net(t_tensor).flatten()
        preprocessed_tensor = preprocess_input(t_tensor, net)
        preprocessed_tensor = preprocessed_tensor.to(device)

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

        if loss.item() < best_total_loss:
            best_total_loss = loss.item()

    param_count = sum(p.numel() for p in net.parameters())
    return best_total_loss, param_count

# Evaluate a child model
def evaluate_child_model(file_path, device):
    model = torch.load(file_path).to(device)
    best_loss, param_count = train_and_evaluate_model(model, device)
    return best_loss, param_count

# Evolution process
def evolve(pool, device, base_dir, num_generations=30):
    for generation in range(num_generations):
        print(f"Generation {generation+1}")
        
        # Generate offspring
        offspring = generate_offspring(pool, num_offspring=3)
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = os.path.join(base_dir, time_stamp)
        os.makedirs(folder_name, exist_ok=True)
        save_offspring_models(offspring, folder_name)
        
        # Evaluate offspring
        for i, child in enumerate(offspring):
            file_path = os.path.join(folder_name, f"nn_new_{i+1}.pt")
            loss, param_count = evaluate_child_model(file_path, device)
            child.fitness.values = (loss,)
            print(f"Child {i+1} | Loss: {loss:.6f} | Params: {param_count}")
        
        # Record generation data
        log_generation_data(generation+1, pool, offspring, base_dir)
        
        # Combine the parent and offspring pools
        combined_pool = pool + offspring
        
        # Select the top 10 models
        pool = tools.selBest(combined_pool, k=10)
        
        # Update the pool with the best loss
        update_pool_best_loss(pool, base_dir)

    return pool

def run_evolution(seed, num_generations=30):
    # Set random seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a unique base directory for Pool_EA with timestamp and seed
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    pool_dir_name = f"Pools_EA_{current_time}_seed_{seed}"
    os.makedirs(pool_dir_name, exist_ok=True)
    
    # Create and save initial models in the time-stamped Pool_EA directory
    create_and_save_models(pool_dir_name)
    
    # Load individuals from files in the time-stamped Pool_EA directory
    pool = load_individuals_from_files(pool_dir_name, "model_pool_loss_history.csv")

    # Evaluate parent models
    evaluate_parent_models(pool, device, pool_dir_name)
    
    # Start the evolution process
    final_pool = evolve(pool, device, pool_dir_name, num_generations=num_generations)
    
    print(f"Evolution completed for seed {seed}.")

if __name__ == "__main__":
    # run si
    seed_list = [100]  # List of different seeds

    for seed in seed_list:
        print(f"Running evolution with seed {seed}")
        run_evolution(seed, num_generations=100)
