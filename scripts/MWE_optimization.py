"""
MWE (Minimum Working Example) for SAASBO Optimization
This script optimizes neural network parameters to minimize heat loss in battery simulation.

Dependencies:
- torch
- numpy
- pandas
- ax-platform
- pybamm
- matplotlib
- scipy
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pybamm
import matplotlib.pyplot as plt
from scipy.integrate import simpson

from ax import Data, Experiment, ParameterType, RangeParameter, SearchSpace
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.result import Ok


# ============================================
# Neural Network Model
# ============================================
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(3)
        self.fc2 = nn.Linear(3, 1, bias=False)
        
    def forward(self, input):
        x = self.fc1(input)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        output = torch.sigmoid(x) * 5  # Scale output to range [0, 5]
        return output


# ============================================
# PyBaMM Simulator
# ============================================
class PyBaMMSimulator:
    def __init__(self):
        self.model = pybamm.lithium_ion.DFN({"thermal": "lumped"})
        self.last_solution = None
        self.results = []
        self.charging_time = 0

    def run_simulation(self, steps):
        solver = pybamm.CasadiSolver(dt_max=200)
        
        for i, step in enumerate(steps):
            self.model = pybamm.lithium_ion.DFN({"thermal": "lumped"})
            self.parameter_values = pybamm.ParameterValues("Chen2020")

            if isinstance(step, dict):
                if "current_function" in step:
                    current_function = step["current_function"]
                    self.parameter_values_new = self.parameter_values
                    self.parameter_values_new["Current function [A]"] = current_function

                t_evals = self.charging_time + step.get("time_evals") if "time_evals" in step else None

                if "termination_voltage" in step:
                    self.parameter_values_new["Upper voltage cut-off [V]"] = step["termination_voltage"]
                
                if i == 0:
                    initial_soc = 0.00001
                    sim = pybamm.Simulation(self.model, parameter_values=self.parameter_values_new, solver=solver)
                    sim.solve(t_eval=t_evals, initial_soc=initial_soc)
                elif self.last_solution:
                    new_model = self.model.set_initial_conditions_from(self.last_solution, inplace=False)
                    sim = pybamm.Simulation(new_model, parameter_values=self.parameter_values_new, solver=solver)
                    sim.solve(t_eval=t_evals)
            else:
                self.experiment = pybamm.Experiment([step])
                if i == 0:
                    initial_soc = 0.00001
                    sim = pybamm.Simulation(self.model, parameter_values=self.parameter_values, experiment=self.experiment, solver=solver)
                    sim.solve(initial_soc=initial_soc)
                elif self.last_solution:
                    new_model = self.model.set_initial_conditions_from(self.last_solution, inplace=False)
                    sim = pybamm.Simulation(new_model, parameter_values=self.parameter_values, experiment=self.experiment, solver=solver)
                    sim.solve()

            self.last_solution = sim.solution
            self.results.append(sim)
            self.charging_time += (sim.solution["Time [s]"].data[-1] - sim.solution["Time [s]"].data[0])

        # Constant voltage step
        final_voltage = round(self.last_solution["Terminal voltage [V]"].data[-1], 3)
        print(f"Final voltage: {final_voltage}")
          
        constant_voltage_step = pybamm.step.voltage(value=final_voltage, termination=["10mA"])
        self.experiment2 = pybamm.Experiment([constant_voltage_step])
        new_model = self.model.set_initial_conditions_from(self.last_solution, inplace=False)
        sim = pybamm.Simulation(new_model, parameter_values=self.parameter_values, experiment=self.experiment2, solver=solver)
        sim.solve()
        self.results.append(sim)
        self.last_solution = sim.solution

        return self.results


class CustomSimulator:
    def __init__(self):
        self.simulator = PyBaMMSimulator()

    def tem_cutoff(self, variables, cutoff_temperature=313.15):
        return cutoff_temperature - variables["X-averaged cell temperature [K]"]

    def create_termination(self, temperature_cutoff=310, voltage_cutoff="4.2V"):
        tem_termination = pybamm.step.CustomTermination(
            name="Temperature cutoff",
            event_function=lambda variables: self.tem_cutoff(variables, temperature_cutoff)
        )
        return [tem_termination, voltage_cutoff]

    def create_step(self, type, value, termination_conditions, t=np.linspace(0, 10000, 5000), function=None, time_cutoff=20000):
        if function is not None:
            assert t is not None, "Time range 't' must be provided for custom current functions."
            function_value = np.column_stack([t, function(t)])
            if type == "current":
                return pybamm.step.current(function_value, termination=termination_conditions, duration=time_cutoff)
            if type == "voltage":
                return pybamm.step.voltage(value=function_value, termination=termination_conditions, duration=time_cutoff)
        else:
            if type == "current":
                return pybamm.step.current(value=value, termination=termination_conditions, duration=time_cutoff)
            if type == "voltage":
                return pybamm.step.voltage(value=value, termination=termination_conditions, duration=time_cutoff)

    def run_simulation_with_steps(self, steps_details):
        steps = []
        for step_detail in steps_details:
            value = step_detail.get('value', None)
            type = step_detail.get('type', None)
            temperature_cutoff = step_detail.get('temperature_cutoff', 315)
            voltage_cutoff = step_detail.get('voltage_cutoff', "4.2V")
            time_cutoff = step_detail.get('time_cutoff', 3600)
            termination_conditions = self.create_termination(temperature_cutoff, voltage_cutoff)
            t = step_detail.get('t', np.linspace(0, 20000, 7000))
            function = step_detail.get('function', None)
            
            step = self.create_step(type=type, value=value, termination_conditions=termination_conditions, t=t, function=function, time_cutoff=time_cutoff)
            steps.append(step)
        
        stages = self.simulator.run_simulation(steps)
        return stages


class SimulationResults:
    def __init__(self, stages, initial_soc=0.00001):
        self.stages = stages
        self.initial_soc = initial_soc
        
        DFN_21700 = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
        param = DFN_21700.param
        parameters = pybamm.ParameterValues("Chen2020")
        self.x_0, self.x_100, self.y_100, self.y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(parameters, param)

    def calculate_heat_at_each_moment(self):
        cumulative_heat = 0
        heat_time_series = [(0, 0)]
        total_time_shift = 0

        for sim in self.stages:
            solution = sim.solution
            heating_data = solution["Total heating [W]"].data
            time_data = solution["Time [s]"].data - solution["Time [s]"].data[0] + total_time_shift
            total_time_shift = time_data[-1]

            for i in range(1, len(time_data)):
                dt = time_data[i] - time_data[i-1]
                dW = heating_data[i-1] * dt
                cumulative_heat += dW
                heat_time_series.append((time_data[i], cumulative_heat))

        return heat_time_series

    def heat_to_reach_target_time(self, target_time):
        total_time_shift = 0
        all_times = np.array([])
        all_heat_generations = np.array([])

        if target_time is None:
            target_time = float('inf')

        for sim in self.stages:
            solution = sim.solution
            time = solution["Time [s]"].data - solution["Time [s]"].data[0] + total_time_shift
            total_time_shift = time[-1]

            all_times = np.concatenate([all_times, time])
            all_heat_generations = np.concatenate([all_heat_generations, solution["Total heating [W]"].data])

            if total_time_shift > target_time:
                break

        all_times = all_times[all_times < target_time]
        all_heat_generations = all_heat_generations[:len(all_times)]

        return all_heat_generations[:-1] if len(all_heat_generations) > 1 else all_heat_generations

    def plot_results(self, save_path):
        total_time_shift = 0
        all_times = []
        all_voltages = []
        all_currents = []
        all_heat_generations = []
        all_temperatures = []

        for sim in self.stages:
            solution = sim.solution
            time = solution["Time [s]"].data - solution["Time [s]"].data[0] + total_time_shift
            total_time_shift = time[-1]

            all_times = np.concatenate([all_times, time])
            all_voltages = np.concatenate([all_voltages, solution["Terminal voltage [V]"].data])
            all_currents = np.concatenate([all_currents, solution["Current [A]"].data])
            all_heat_generations = np.concatenate([all_heat_generations, solution["Total heating [W]"].data])
            all_temperatures = np.concatenate([all_temperatures, solution["Cell temperature [K]"].data[-1]])

        plt.figure(figsize=(12, 16))

        plt.subplot(4, 1, 1)
        plt.plot(all_times, all_temperatures, label='Temperature')
        plt.ylabel('Temperature [K]')
        plt.title('Temperature Over Stages')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(all_times, all_heat_generations, label='Heat Generation')
        plt.ylabel('Total Heat Generation [W]')
        plt.title('Heat Generation Over Stages')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(all_times, all_voltages, label='Voltage')
        plt.ylabel('Terminal Voltage [V]')
        plt.title('Voltage Over Stages')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(all_times, all_currents, label='Current')
        plt.xlabel('Time [s]')
        plt.ylabel('Current [A]')
        plt.title('Current Over Stages')
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        plt.close()

    def show_results(self, save_path=None):
        self.plot_results(save_path)


# ============================================
# Optimization Metric
# ============================================
class HeatLossMetric(Metric):
    def __init__(self, name: str = "heat_loss", model=None, device=None, result_folder=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.device = device
        self.result_folder = result_folder
        self.best_loss = float('inf')
        self.results_buffer = []

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            heat_loss = self.objective(params)
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": heat_loss,
                "sem": 0.0,
            })
        df = pd.DataFrame.from_records(records)
        print(f"Trial {trial.index} data fetched: {df}")
        return Ok(Data(df=df))

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
        """MSE loss between heat series and target heat"""
        if isinstance(heat_series, (list, np.ndarray)):
            heat_array = np.asarray(heat_series, dtype=np.float64)
            if heat_array.size == 0:
                return 1e6
            return np.mean((heat_array - target_heat) ** 2)
        return 1e6

    def record_result(self, params, heat_loss, is_optimal):
        self.results_buffer.append({
            'params': params,
            'heat_loss': heat_loss,
            'is_optimal': 'Optimal' if is_optimal else 'No'
        })

    def flush_results(self):
        """Write all buffered results to CSV at once"""
        if not self.results_buffer:
            return
        
        results_df = pd.DataFrame(self.results_buffer)
        
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        
        results_file = os.path.join(self.result_folder, 'optimization_results.csv')
        
        if os.path.exists(results_file):
            results_df.to_csv(results_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(results_file, index=False, header=True)
        
        self.results_buffer.clear()

    def objective(self, params):
        """Objective function: compute heat loss from simulation"""
        # Extract parameters
        param_list = [params[f'{name}_{i}'] 
                      for name, param in self.model.named_parameters() 
                      for i in range(param.numel())]
        print(f"Parameters: {param_list}")
        
        model = self.model.to(self.device)
        self.apply_parameters(model, param_list)
        
        # Generate time series input
        t_values = np.linspace(0, 3600, 1000).reshape(-1, 1)
        t_tensor = torch.tensor(t_values, dtype=torch.float32).to(self.device)
        
        try:
            # Forward pass through neural network
            output = model(t_tensor).detach().cpu().numpy()
            t_values_update = np.linspace(0, 3600, 1000).reshape(-1, 1)
            output_update = -output
            data = np.column_stack((t_values_update, output_update))
            
            # Run simulation
            steps_details = [{
                'type': "current", 
                'value': data, 
                'temperature_cutoff': 315, 
                'voltage_cutoff': "4.2V", 
                'duration': 3600
            }]
            simulator = CustomSimulator()
            stages = simulator.run_simulation_with_steps(steps_details)
            results = SimulationResults(stages)

            # Compute loss
            heat = results.heat_to_reach_target_time(3600)
            target_heat = 0.4
            heat_loss = self.loss_function(heat, target_heat)

            # Track best loss
            is_optimal = heat_loss < self.best_loss
            if is_optimal:
                self.best_loss = heat_loss

            # Buffer result
            self.record_result(params, heat_loss, is_optimal)

            # Save plot for optimal results
            if is_optimal:
                save_path = os.path.join(self.result_folder, "optimal_sim_plot.png")
                results.show_results(save_path)

            return heat_loss

        except Exception as e:
            print(f"Error during simulation: {e}")
            return 1e6


# ============================================
# Main Optimization Function
# ============================================
def run_optimization(result_folder="./results", random_seed=None):
    """
    Run SAASBO optimization
    
    Args:
        result_folder: Folder to save results
        random_seed: Random seed for reproducibility (None for true randomness)
    """
    # Set random seeds
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        print(f"Random seed: {random_seed}")
    else:
        print("Using true randomness (no fixed seed)")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create result folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Initialize model
    model = NeuralNetwork().to(device)

    # Generate search space from model parameters
    def generate_search_space(model):
        space = []
        for name, param in model.named_parameters():
            num_params = param.numel()
            space += [
                RangeParameter(
                    name=f'{name}_{i}', 
                    parameter_type=ParameterType.FLOAT, 
                    lower=-1.0, 
                    upper=1.0
                )
                for i in range(num_params)
            ]
        print(f"Search space dimension: {len(space)}")
        return SearchSpace(parameters=space)

    search_space = generate_search_space(model)

    # Define optimization config
    heat_loss_metric = HeatLossMetric(
        name="heat_loss", 
        model=NeuralNetwork(), 
        device=device, 
        result_folder=result_folder
    )
    
    optimization_config = OptimizationConfig(
        objective=Objective(metric=heat_loss_metric, minimize=True)
    )

    # SAASBO hyperparameters (consistent with original)
    N_INIT = 30      # Number of initial Sobol points
    BATCH_SIZE = 3   # Batch size for SAASBO
    N_BATCHES = 60   # Number of SAASBO batches

    print(f"Total evaluations: {N_INIT + N_BATCHES * BATCH_SIZE}")

    # Setup experiment
    experiment = Experiment(
        name="saasbo_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )

    # Phase 1: Initial Sobol sampling
    print("\n=== Phase 1: Sobol Initialization ===")
    sobol = Models.SOBOL(search_space=experiment.search_space)
    for i in range(N_INIT):
        trial = experiment.new_trial(sobol.gen(1))
        trial.run()
        
        if (i + 1) % 10 == 0:
            heat_loss_metric.flush_results()
            print(f"Completed {i + 1}/{N_INIT} Sobol trials")
    
    heat_loss_metric.flush_results()

    # Get initial data
    data = experiment.fetch_data()
    if data.df.empty:
        print("Error: Initial data from Sobol trials is empty!")
        return None, None
    
    print(f"Initial best: {data.df['mean'].min():.6f}")

    # Phase 2: SAASBO optimization
    print("\n=== Phase 2: SAASBO Optimization ===")
    for i in range(N_BATCHES):
        model_saasbo = Models.SAASBO(experiment=experiment, data=data)
        generator_run = model_saasbo.gen(BATCH_SIZE)
        trial = experiment.new_batch_trial(generator_run=generator_run)
        trial.run()
        new_data = trial.fetch_data()
        data = Data.from_multiple_data([data, new_data])

        new_value = new_data.df["mean"].min()
        print(f"Batch {i+1}/{N_BATCHES}, Best in batch: {new_value:.6f}, Best overall: {data.df['mean'].min():.6f}")
        
        if (i + 1) % 10 == 0:
            heat_loss_metric.flush_results()
    
    heat_loss_metric.flush_results()

    # Save final results
    best_heat_loss = data.df["mean"].min()
    print(f"\n=== Optimization Complete ===")
    print(f"Best heat loss: {best_heat_loss:.6f}")

    best_loss_file = os.path.join(result_folder, "best_loss.txt")
    with open(best_loss_file, "w") as f:
        f.write(f"{best_heat_loss}")
    print(f"Best loss saved to {best_loss_file}")

    return result_folder, best_heat_loss


# ============================================
# Entry Point
# ============================================
if __name__ == '__main__':
    # Run optimization with a fixed seed for reproducibility
    # Set random_seed=None for true randomness
    result_folder, best_loss = run_optimization(
        result_folder="./experiments/MWE_results",
        random_seed=42
    )
    
    if result_folder:
        print(f"\nResults saved to: {result_folder}")
        print(f"Best loss achieved: {best_loss}")
