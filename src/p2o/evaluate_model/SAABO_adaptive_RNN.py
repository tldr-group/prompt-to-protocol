import os
import sys
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from ax import Data, Experiment, ParameterType, RangeParameter, SearchSpace
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from src.p2o.simulation.sim_p2o_c3 import BatteryCyclingExperiment
from ax.utils.common.result import Ok
import csv
import multiprocessing as mp
from matplotlib.animation import FuncAnimation
import datetime
import os
import importlib
from functools import partial
from src.tools.utils import *
import pybamm


class TotalLossMetric(Metric):
    def __init__(self, name: str = "total_loss", model=None, nn_in_pybamm=None, device=None, result_folder=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.device = device
        self.result_folder = result_folder
        self.nn_in_pybamm = nn_in_pybamm
        self.h_prev = None   

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            total_loss = self.objective(params)
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": total_loss,
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

    def loss_function(self, SOH):

        if SOH is None:
            SOH = 1e-6

        # If SOH is smaller than 0.6, return a large penalty
        if SOH < 0.6:
            return 1e6

        normalized = (SOH - 0.6) / (1.0 - 0.6)
        soh_loss = -np.log(normalized)
        total_loss = soh_loss
        return total_loss
    
    def record_result(self, params, total_loss, SOH, is_optimal):
        result_data = {
            'params': params,
            'total_loss': total_loss,
            'SOH': SOH,
            'is_optimal': 'Optimal' if is_optimal else 'No'
        }
        results_df = pd.DataFrame([result_data])
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        results_file = os.path.join(self.result_folder, 'optimization_results.csv')
        if not os.path.exists(results_file):
            results_df.to_csv(results_file, index=False, header=True)
        else:
            results_df.to_csv(results_file, mode='a', header=False, index=False)
        
        pth_file = os.path.join(self.result_folder, 'all_params.pth')
        try:
            if os.path.exists(pth_file):
                stored_params = torch.load(pth_file)
            else:
                stored_params = []
            stored_params.append(params)
            torch.save(stored_params, pth_file)
        except Exception as e:
            print(f"Error saving parameters to .pth file: {e}")
    
    def get_fc_weights(self, model: nn.Module):	
        """Return (Ws, bs) lists for *all* nn.Linear layers in *forward order*.

        The order is determined by a forward trace via ``torch.fx``, ensuring that
        even with branches / loops the weights align with actual execution.
        """
        import torch.fx as fx

        dummy = torch.randn(1, 2)  # shape matches network input
        gm = fx.symbolic_trace(model)

        Ws, bs = [], []
        lin_modules = dict(model.named_modules())
        for node in gm.graph.nodes:
            if node.op == "call_module":
                mod = lin_modules[node.target]
                if isinstance(mod, nn.Linear):
                    Ws.append(mod.weight.detach().cpu().numpy())
                    if mod.bias is not None:  
                        bs.append(mod.bias.detach().cpu().numpy())
        return Ws, bs
    
    def is_rnn_model(self):
        """
        Detect if model is RNN type.
        Returns:
            bool: True if RNN (4 parameters), False if MLP (3 parameters)
        """
        import inspect
        nn_func_params = inspect.signature(self.nn_in_pybamm).parameters
        return len(nn_func_params) == 4
    
    @staticmethod
    def adaptive_current_core(vars_dict, self_ref, Ws, bs):
        V   = vars_dict["Voltage [V]"]
        T   = vars_dict["Volume-averaged cell temperature [K]"]
        soc = vars_dict["SoC"]

        Vn = 2 * ((V - 3.0) / (4.2 - 3.0)) - 1
        Tn = 2 * ((T - 308.15) / (13)) - 1
        Sn = 2 * (soc - 0.5)

        X = [pybamm.Scalar(10) * Vn,
            pybamm.Scalar(10) * Tn,
            pybamm.Scalar(10) * Sn]

        # Auto-detect model type and call corresponding function
        if self_ref.is_rnn_model():  # RNN: (X, H_prev, Ws, bs)
            if self_ref.h_prev is None:
                self_ref.h_prev = [pybamm.Scalar(0) for _ in range(len(Ws[0]))]
            
            y_t, h_t = self_ref.nn_in_pybamm(X, self_ref.h_prev, Ws, bs)
            self_ref.h_prev = h_t
            return y_t
        else:  # MLP: (X, Ws, bs)
            return self_ref.nn_in_pybamm(X, Ws, bs)
    

    def make_pybamm_current(self, Ws, bs):
        # Auto-detect model type and initialize (only RNN needs hidden state)
        if self.is_rnn_model():
            if self.h_prev is None:
                self.h_prev = [pybamm.Scalar(0) for _ in range(len(Ws[0]))]

        from functools import partial
        return partial(
            self.adaptive_current_core,
            self_ref=self,
            Ws=Ws,
            bs=bs,
        )

    def objective(self, params):
        param_list = [params[f'{name}_{i}'] for name, param in self.model.named_parameters() for i in range(param.numel())]
        print(f"Parameters: {param_list}")
        model = self.model.to(self.device)
        self.apply_parameters(model, param_list)
        Ws, bs = self.get_fc_weights(model)
        charging_current = self.make_pybamm_current(Ws, bs)
        
        try:
            experiment = BatteryCyclingExperiment(num_cycles=100, SOC=0.9, adaptive_current=charging_current)
            final_solution, soh_values = experiment.run_cycles()
            SOH = soh_values[-1]
            votage_max_list = final_solution["Voltage [V]"].data
                
            total_loss = self.loss_function(SOH)

            # record results
            self.record_result(params, total_loss, SOH, is_optimal=False)

            is_optimal = False

            total_loss_short = round(total_loss, 6)
            results_df = pd.read_csv(os.path.join(self.result_folder, 'optimization_results.csv'))
            min_loss = round(results_df['total_loss'].min(), 6) 

            if total_loss_short <= min_loss:
                is_optimal = True
            
            if is_optimal:
                save_path = os.path.join(self.result_folder)
                experiment.plot_results(final_solution, soh_values, save_path)

            return total_loss

        except Exception as e:
            print(f"Error during simulation: {e}")
            return 1e6  # Return a large number if an error occurs

def evaluate_model(module_name, result_folder):
    # Import the neural network module and set up the device
    print(f"Starting evaluation for: {module_name}")
    nn_module = importlib.import_module(module_name)
    NeuralNetwork = nn_module.NeuralNetwork
    nn_in_pybamm = nn_module.nn_in_pybamm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model imported successfully.")
    
    def mark_optimal_result(result_folder):
        results_file = os.path.join(result_folder, 'optimization_results.csv')
        results_df = pd.read_csv(results_file)
        min_loss_idx = results_df['total_loss'].idxmin()
        results_df.loc[min_loss_idx, 'is_optimal'] = 'Optimal'
        results_df.to_csv(results_file, index=False)

    def record_nn_result(best_total_loss, result_folder):
        csv_file = 'nn_optimization_results.csv'
        is_optimal = False

        # Check if the CSV file exists
        if not os.path.exists(csv_file):
            # Create a new CSV file and write the header
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Folder', 'Best_Total_Loss', 'Is_Optimal_So_Far'])

        # Read the current best total loss from the CSV file
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            losses = [float(row['Best_Total_Loss']) for row in reader]
            if losses:
                min_loss = min(losses)
                if best_total_loss < min_loss:
                    is_optimal = True
            else:
                is_optimal = True

        # Append the new result to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([result_folder, best_total_loss, is_optimal])


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
            metric=TotalLossMetric(
                name="total_loss", 
                model=NeuralNetwork(), 
                nn_in_pybamm=nn_in_pybamm, 
                device=device, 
                result_folder=result_folder
            ),
            minimize=True,
        )
    )


    N_INIT = 30  
    BATCH_SIZE = 3
    N_BATCHES = 60


    # N_INIT = 1 # Reduced for debugging purposes
    # BATCH_SIZE = 2
    # N_BATCHES = 1


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

    # Ensure we have initial data before running SAASBO
    try:
        data = experiment.fetch_data()
        print(f"Initial data: {data.df}")
    except Exception as e:
        print(f"Exception occurred while fetching data: {e}")
        return

    if data.df.empty:
        print("Initial data from Sobol trials is empty. Check the objective function.")
        
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

        print("Reading the best total loss from CSV to ensure consistency.")
        results_df = pd.read_csv(os.path.join(result_folder, 'optimization_results.csv'))
        best_total_loss = results_df['total_loss'].min()
        print("CSV-based best total loss:", best_total_loss)

        # Record the best model
        mark_optimal_result(result_folder)  
        record_nn_result(best_total_loss, result_folder)  

        best_loss_file = os.path.join(result_folder, "best_loss.txt")
        try:
            with open(best_loss_file, "w") as f:
                f.write(f"{best_total_loss}")
            print(f"Best loss saved to {best_loss_file}")
        except Exception as e:
            print(f"Error while writing best loss to file: {e}")

        return result_folder, best_total_loss