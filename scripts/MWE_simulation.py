import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn as nn
from ax.core.metric import Metric
from src.p2o.simulation.sim_p2o_c1_c2 import CustomSimulator, SimulationResults
from src.tools.show_searching_space import show_searching_space

# Define the Neural Network Model
     
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
        output = torch.sigmoid(x) * 10  # Scale output to range [0, 10]
        return output


# Define the Custom Metric for Total Loss
class TotalLossMetric(Metric):
    def __init__(self, name: str = "total_loss", model=None, device=None, result_folder=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.device = device
        self.result_folder = result_folder

    def apply_parameters(self, model, param_list):
        """Apply a list of parameters to the model."""
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
                loss = 1e6
            else:
                loss = np.mean((heat_array - target_heat) ** 2)
        else:
            loss = 1e6  # Assign a large number if heat_series is not iterable
        return loss
    
    def show_results(self, save_path):
        energy_efficiency = self.energy_efficiency()
        total_heat_generated = self.total_heat_generated()
        total_charging_time = self.total_charging_time()
        calculated_soc_at_each_moment = self.calculate_soc_at_each_moment()
        estimated_soc = calculated_soc_at_each_moment[-1][-1]
        final_voltage = self.final_terminal_voltage()
        termination_reasons = self.termination_reasons()
        time_to_reach_target_soc = self.time_to_reach_target_soc(80)
        energy_to_reach_target_soc = self.heat_to_reach_target_soc(80)

        print(f"Time to reach 80% SOC: {time_to_reach_target_soc} s")
        print(f"Heat generated to reach 80% SOC: {energy_to_reach_target_soc} J")
        print(f"Total heat generated: {total_heat_generated} J")
        print(f"Total charging time: {total_charging_time} s")
        print(f"Estimated final SOC: {estimated_soc:.2f}%")
        print(f"Final terminal voltage: {final_voltage} V")
        print(f"Termination reasons: {termination_reasons}")
        # print(f"Energy efficiency: {energy_efficiency:.2f}")
        print(f"Max temperature: {self.calculate_max_temperature()} K")

        # Call the plot function to visualize the results
        self.plot_results(save_path)

    def objective(self, params):
        """Objective function to compute total loss given parameters."""
        # Extract parameter values from the params dictionary
        param_list = [params[f'{name}_{i}'] 
                      for name, param in self.model.named_parameters() 
                      for i in range(param.numel())]
        print(f"Applying Parameters: {param_list}")
        model = self.model.to(self.device)

        # Apply parameters to the model
        self.apply_parameters(self.model, param_list)
        self.model.to(self.device)

        # Example input data (you can modify this as needed)
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

            charging_time = results.time_to_reach_target_soc(80)
            temperature = results.calculate_max_temperature()
            voltage = results.final_terminal_voltage()
            SOC = results.estimated_soc()
            if charging_time is not None:
                time_to_reach_target_voltage = results.time_to_reach_target_voltage(4.2)
                heat = results.heat_to_reach_target_time(time_to_reach_target_voltage)
            else:
                heat = 1e6
                charging_time = 1e6
                
            target_heat = 0.7
            total_loss = self.loss_function(heat, target_heat)

            # record results
            save_path = os.path.join(self.result_folder, "sim_plot.png")
            results.show_results(save_path)
            return total_loss

        except Exception as e:
            print(f"Error during simulation: {e}")
            return 1e6  # Return a large number if an error occurs

def print_model_parameters(model):
    """Print the number and types of parameters in the model."""
    print("Model Parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        print(f" - {name}: {num} parameters")
    print(f"Total number of parameters: {total_params}\n")

def generate_params(model, custom_params=None, dtype=np.float32):
    """
    Generate a dictionary of parameters.
    - If custom_params is provided, use those for specified layers.
    - Otherwise, initialize parameters randomly.
    """
    params = {}
    for name, param in model.named_parameters():
        num_params = param.numel()
        if custom_params and name in custom_params:
            # Use custom parameters if provided
            param_values = np.array(custom_params[name], dtype=dtype)
            if len(param_values) != num_params:
                raise ValueError(f"Custom parameter list for {name} must have {num_params} values.")
            params.update({f'{name}_{i}': param_values[i] for i in range(num_params)})
        else:
            # Initialize parameters randomly
            param_values = np.random.uniform(-1.0, 1.0, num_params).astype(dtype)
            params.update({f'{name}_{i}': param_values[i] for i in range(num_params)})
    return params

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Initialize the model
    model = NeuralNetwork().to(device)
    
    # Print model parameters
    print_model_parameters(model)

    # Define result folder
    result_folder = "./experiments/results_MWE_c1_c2"
    # Create result folder if it doesn't exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"Directory '{result_folder}' created.")
    else:
        print(f"Directory '{result_folder}' already exists.")

    # Initialize the metric
    metric = TotalLossMetric(name="total_loss", model=model, device=device, result_folder=result_folder)

    # Define custom parameters (you can modify these as needed)
    # Ensure that the number of custom parameters matches the model's requirements
    custom_params = {
        'fc1.weight': [0.5789] * model.fc1.weight.numel(),   # 3 parameters for fc1.weight
        'bn1.weight': [0.23] * model.bn1.weight.numel(),      # 3 parameters for bn1.weight
        'bn1.bias': [0.0] * model.bn1.bias.numel(),          # 3 parameters for bn1.bias
        'fc2.weight': [-0.5] * model.fc2.weight.numel(),     # 3 parameters for fc2.weight
    }
# "{'fc1.weight_0': -0.3918332233669527, 'fc1.weight_1': -0.4542044132586246, 'fc1.weight_2': 0.4878317371729606, 'bn1.weight_0': 0.6012526020297413, 'bn1.weight_1': -0.06709455144747745, 'bn1.weight_2': 1.0, 'bn1.bias_0': -0.21779405384152328, 'bn1.bias_1': -1.0, 'bn1.bias_2': -0.4331677702387384, 'fc2.weight_0': -0.39509981494936286, 'fc2.weight_1': -0.2616128353313648, 'fc2.weight_2': 0.049227622882867195}",0.8090520812072677,0.0008090520812072677,3166.3662297615324,310.832409976911,4.199999999999999,99.93475733581346,No

    # Generate parameters, allowing some to be custom and others random
    params = generate_params(model, custom_params=custom_params)

    # Print the generated parameters
    print("Generated Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print()

    # Compute the total loss using the objective function
    total_loss = metric.objective(params)
    print(f"\nTotal Loss: {total_loss}")

if __name__ == "__main__":
    main()
