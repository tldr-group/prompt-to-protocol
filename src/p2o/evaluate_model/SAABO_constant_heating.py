import os
import sys
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from ax import Data, Experiment, ParameterType, RangeParameter, SearchSpace
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from src.p2o.simulation.sim_p2o_c1_c2 import CustomSimulator, SimulationResults
from ax.utils.common.result import Ok
import csv
import multiprocessing as mp

from src.tools.utils import *


class HeatLossMetric(Metric):
    def __init__(self, name: str = "heat_loss", model=None, device=None, result_folder=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.device = device
        self.result_folder = result_folder
        self.best_loss = float('inf')  # Track best loss in memory
        self.results_buffer = []  # Buffer results to write in batch

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            heat_loss = self.objective(params)
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": heat_loss,
                    "sem": 0.0,
                }
            )
        df = pd.DataFrame.from_records(records)
        print(f"Trial {trial.index} data fetched: {df}")  # Debug information
        data = Data(df=pd.DataFrame.from_records(records))
        return Ok(data)

    def is_available_while_running(self) -> bool:
        return True

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
        # Buffer results in memory instead of writing immediately
        result_data = {
            'params': params,
            'heat_loss': heat_loss,
            'is_optimal': 'Optimal' if is_optimal else 'No'
        }
        self.results_buffer.append(result_data)
    
    def flush_results(self):
        """Write all buffered results to CSV at once"""
        if not self.results_buffer:
            return
        
        columns = ['params', 'heat_loss', 'is_optimal']
        results_df = pd.DataFrame(self.results_buffer, columns=columns)
        
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        
        results_file = os.path.join(self.result_folder, 'optimization_results.csv')
        
        if os.path.exists(results_file):
            results_df.to_csv(results_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(results_file, index=False, header=True)
        
        self.results_buffer.clear()
            

    def objective(self, params):
        param_list = [params[f'{name}_{i}'] for name, param in self.model.named_parameters() for i in range(param.numel())]
        print(f"Parameters: {param_list}")
        model = self.model.to(self.device)
        self.apply_parameters(model, param_list)
        
        t_values = np.linspace(0, 3600, 1000).reshape(-1, 1)
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

            # Track best loss in memory (much faster than reading from file)
            is_optimal = heat_loss < self.best_loss
            if is_optimal:
                self.best_loss = heat_loss

            # Buffer result in memory
            self.record_result(params, heat_loss, is_optimal=is_optimal)

            # Only save plot for truly optimal results (and not too frequently)
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
    nn_module = importlib.import_module(module_name)
    NeuralNetwork = nn_module.NeuralNetwork
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model imported successfully.")

    # Some tools to record the result
    
    def mark_optimal_result(result_folder):
        results_file = os.path.join(result_folder, 'optimization_results.csv')
        results_df = pd.read_csv(results_file)
        min_loss_idx = results_df['heat_loss'].idxmin()
        results_df.loc[min_loss_idx, 'is_optimal'] = 'Optimal'
        results_df.to_csv(results_file, index=False)

    def record_nn_result(best_heat_loss, result_folder):
        csv_file = 'nn_optimization_results.csv'
        is_optimal = False

        # Check if the CSV file exists
        if not os.path.exists(csv_file):
            # Create a new CSV file and write the header
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Folder', 'Best_Heat_Loss', 'Is_Optimal_So_Far'])

        # Read the current best heat loss from the CSV file
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            losses = [float(row['Best_Heat_Loss']) for row in reader]
            if losses:
                min_loss = min(losses)
                if best_heat_loss < min_loss:
                    is_optimal = True
            else:
                is_optimal = True

        # Append the new result to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([result_folder, best_heat_loss, is_optimal])


    # Optimization tools
    # Function to generate the search space
    def generate_search_space(model):
        space = []
        for name, param in model.named_parameters():
            num_params = param.numel()
            space += [
                RangeParameter(
                    name=f'{name}_{i}', parameter_type=ParameterType.FLOAT, lower=-1.0, upper=1.0
                )
                for i in range(num_params)
            ]
        print(f"Generated search space: {space}")
        return SearchSpace(parameters=space)

    # Generate search space
    search_space = generate_search_space(NeuralNetwork().to(device))

    # Define the optimization config
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=HeatLossMetric(name="heat_loss", model=NeuralNetwork(), device=device, result_folder=result_folder),
            minimize=True,
        )
    )
    
    # Get a reference to the metric for later use
    heat_loss_metric = optimization_config.objective.metric

    N_INIT = 30  # Reduced for debugging purposes
    BATCH_SIZE = 3
    N_BATCHES = 60

    print(f"Doing {N_INIT + N_BATCHES * BATCH_SIZE} evaluations")

    # Set up the experiment
    experiment = Experiment(
        name="saasbo_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )

    # Initial Sobol points
    sobol = Models.SOBOL(search_space=experiment.search_space)
    for i in range(N_INIT):
        trial = experiment.new_trial(sobol.gen(1))
        trial.run()
        
        # Flush results every 10 trials to avoid memory buildup
        if (i + 1) % 10 == 0:
            heat_loss_metric.flush_results()
            print(f"Flushed results after {i + 1} trials")
    
    # Flush remaining results
    heat_loss_metric.flush_results()

    # Ensure we have initial data before running SAASBO
    try:
        data = experiment.fetch_data()
        print(f"Initial data: {data.df}")
    except Exception as e:
        print(f"Exception occurred while fetching data: {e}")

    if data.df.empty:
        print("Initial data from Sobol trials is empty. Check the objective function.")
        return
    else:
        print("Data after initial Sobol points:")
        print(data.df)  # Print the DataFrame to see its content

        # Run SAASBO
        for i in range(N_BATCHES):
            model = Models.SAASBO(experiment=experiment, data=data)
            generator_run = model.gen(BATCH_SIZE)
            trial = experiment.new_batch_trial(generator_run=generator_run)
            trial.run()
            new_data = trial.fetch_data()
            data = Data.from_multiple_data([data, new_data])

            new_value = new_data.df["mean"].min()
            print(
                f"Iteration: {i}, Best in iteration {new_value:.3f}, Best so far: {data.df['mean'].min():.3f}"
            )
            
            # Flush results periodically
            if (i + 1) % 10 == 0:
                heat_loss_metric.flush_results()
                print(f"Flushed results after batch {i + 1}")
        
        # Flush any remaining results
        heat_loss_metric.flush_results()

        best_heat_loss = data.df["mean"].min()
        print("Best objective value: ", best_heat_loss)

        # Record the best model
        mark_optimal_result(result_folder)  
        record_nn_result(best_heat_loss, result_folder)  

        best_loss_file = os.path.join(result_folder, "best_loss.txt")
        try:
            with open(best_loss_file, "w") as f:
                f.write(f"{best_heat_loss}")
            print(f"Best loss saved to {best_loss_file}")
        except Exception as e:
            print(f"Error while writing best loss to file: {e}")

        return result_folder, best_heat_loss

        
