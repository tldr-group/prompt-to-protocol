import inspect
import os
import torch
import numpy as np
import pandas as pd
from simulation import CustomSimulator, SimulationResults
import csv
import re
from datetime import datetime
from show_searching_space import show_searching_space
import multiprocessing as mp
import time
from utils import *

class HeatLossMetric:
    def __init__(self, model=None, device=None, result_folder=None):
        self.model = model
        self.device = device
        self.result_folder = result_folder

    def apply_parameters(self, model, param_list):
        param_dict = {}
        idx = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            param_dict[name] = torch.tensor(param_list[idx:idx + num_params], device=self.device).reshape(param.shape)
            idx += num_params

        for name, param in model.named_parameters():
            param.data = param_dict[name]

    def loss_function(self, heat_series, target_heat):
        if isinstance(heat_series, (list, np.ndarray)):
            heat_array = np.asarray(heat_series, dtype=np.float64)
            if heat_array.size == 0:
                heat_loss = 1e6
            else:
                heat_loss = np.mean((heat_array - target_heat) ** 2)
        else:
            heat_loss = 1e6  # Assign a large number if heat_series is not iterable
        return heat_loss
    
    def record_result(self, params, heat_loss, is_optimal):
        result_data = {
            'params': params,
            'heat_loss': heat_loss,
            'is_optimal': 'Optimal' if is_optimal else 'No'
        }
        columns = ['params', 'heat_loss', 'is_optimal']
        results_df = pd.DataFrame([result_data], columns=columns)
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        results_file = os.path.join(self.result_folder, 'optimization_results.csv')
        if os.path.exists(results_file):
            try:
                existing_df = pd.read_csv(results_file)
                existing_df = existing_df.reindex(columns=columns)
            except Exception:
                existing_df = pd.DataFrame(columns=columns)
            existing_df.to_csv(results_file, index=False, header=True)
            results_df.to_csv(results_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(results_file, index=False, header=True)

    def objective(self, params):
        param_list = [params[f'{name}_{i}'] for name, param in self.model.named_parameters() for i in range(param.numel())]
        print(f"Parameters: {param_list}")
        model = self.model.to(self.device)
        self.apply_parameters(model, param_list)
        
        t_values = np.linspace(-10, 10, 1000).reshape(-1, 1)
        t_tensor = torch.tensor(t_values, dtype=torch.float32).to(self.device)
        
        try:
            output = model(t_tensor).detach().cpu().numpy()
            t_values_update = np.linspace(0, 3600, 1000).reshape(-1, 1)
            output_update = -output
            data = np.column_stack((t_values_update, output_update))
            steps_details = [
                {'type': "current", 'value': data, 'temperature_cutoff': 315, 'voltage_cutoff': "4.2V", 'duration': 3600}
            ]
            simulator = CustomSimulator()
            stages = simulator.run_simulation_with_steps(steps_details)
            results = SimulationResults(stages)

            heat = results.heat_to_reach_target_time(3600)
            target_heat = 0.4
            heat_loss = self.loss_function(heat, target_heat)

            results_file = os.path.join(self.result_folder, 'optimization_results.csv')
            previous_best = None
            if os.path.exists(results_file):
                try:
                    existing_df = pd.read_csv(results_file)
                    if 'heat_loss' in existing_df.columns and not existing_df.empty:
                        previous_best = existing_df['heat_loss'].min()
                except Exception:
                    previous_best = None
            is_optimal = previous_best is None or heat_loss <= previous_best

            # record results
            self.record_result(params, heat_loss, is_optimal=is_optimal)

            if is_optimal:
                save_path = os.path.join(self.result_folder, "optimal_sim_plot.png")
                results.show_results(save_path)

            return heat_loss

        except Exception as e:
            print(f"Error during simulation: {e}")
            return 1e6  # Return a large number if an error occurs

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
    
    # Import the neural network module and set up the device
    nn_module = __import__(module_name)
    NeuralNetwork = nn_module.NeuralNetwork
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def mark_optimal_result(result_folder):
        results_file = os.path.join(result_folder, 'optimization_results.csv')
        results_df = pd.read_csv(results_file)
        min_loss_idx = results_df['heat_loss'].idxmin()
        results_df.loc[min_loss_idx, 'is_optimal'] = 'Optimal'
        results_df.to_csv(results_file, index=False)

    def record_nn_result(best_heat_loss, result_folder):
        csv_file = 'nn_optimization_results.csv'
        is_optimal = False

        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Folder', 'Best_Heat_Loss', 'Is_Optimal_So_Far'])

        with open(csv_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            losses = [float(row['Best_Heat_Loss']) for row in reader]
            if losses:
                min_loss = min(losses)
                if best_heat_loss < min_loss:
                    is_optimal = True
            else:
                is_optimal = True

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([result_folder, best_heat_loss, is_optimal])

    search_space = generate_search_space(NeuralNetwork().to(device))

    metric = HeatLossMetric(model=NeuralNetwork(), device=device, result_folder=result_folder)
    N_SAMPLES = 200

    best_params = None
    best_loss = float('inf')

    for i in range(N_SAMPLES):
        params = {}
        np.random.seed(int(time.time() * 1000000000) % (2**32 - 1))
        for name, param in NeuralNetwork().named_parameters():
            num_params = param.numel()
            for j in range(num_params):
                params[f'{name}_{j}'] = np.random.uniform(-1.0, 1.0)
        loss = metric.objective(params)
        if loss < best_loss:
            best_loss = loss
            best_params = params
        print(f"Sample {i+1}/{N_SAMPLES}, Loss: {loss:.6f}, Best Loss: {best_loss:.6f}")

    print("Best parameters found: ", best_params)
    print("Best heat loss: ", best_loss)

    mark_optimal_result(result_folder)  
    record_nn_result(best_loss, result_folder)  

    best_loss_file = os.path.join(result_folder, "best_loss.txt")
    try:
        with open(best_loss_file, "w") as f:
            f.write(f"{best_loss}")
        print(f"Best loss saved to {best_loss_file}")
    except Exception as e:
        print(f"Error while writing best loss to file: {e}")

    return result_folder, best_loss


