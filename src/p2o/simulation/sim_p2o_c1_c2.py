import pybamm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps 
import os

class PyBaMMSimulator:
    def __init__(self, model_type="lithium_ion"):
        self.model_type = model_type
        self.model = pybamm.lithium_ion.DFN({"thermal": "lumped"})
        self.last_solution = None
        self.results = []
        self.charging_time = 0

    def run_simulation(self, steps):
        # fast_solver = pybamm.CasadiSolver(
        # mode="fast", extra_options_setup={"max_num_steps": 1000}
        # )
        # safe_solver = pybamm.CasadiSolver(mode="safe")
        solver = pybamm.CasadiSolver(dt_max=200)
        # solver = pybamm.IDAKLUSolver(rtol=1e-9, atol=1e-9)
        
        for i, step in enumerate(steps):
            self.model = pybamm.lithium_ion.DFN({"thermal": "lumped"})
            self.parameter_values = pybamm.ParameterValues("Chen2020")

            if isinstance(step, dict):
                if "current_function" in step:
                    current_function = step["current_function"]
                    self.parameter_values_new = self.parameter_values
                    self.parameter_values_new["Current function [A]"] = current_function

                if "time_evals" in step:
                    t_evals = self.charging_time + step["time_evals"]
                    
                else:
                    t_evals = None

                if "termination_voltage" in step:
                    termination_voltage = step["termination_voltage"]
                    self.parameter_values_new["Upper voltage cut-off [V]"] = termination_voltage
                
                if i == 0:
                    initial_soc = 0.00001
                    sim = pybamm.Simulation(self.model, parameter_values=self.parameter_values_new, solver=solver)
                    sim.solve(t_eval=t_evals,initial_soc=initial_soc)
                
                elif self.last_solution:
                # Set initial conditions from the last solution
                    new_model = self.model.set_initial_conditions_from(self.last_solution, inplace=False)
                    sim = pybamm.Simulation(new_model, parameter_values=self.parameter_values_new,solver=solver)
                    sim.solve(t_eval=t_evals)

            else:
                # Constant current/voltage - use experiment
                self.experiment = pybamm.Experiment([step])
                if i == 0:
                    initial_soc = 0.00001
                    sim = pybamm.Simulation(self.model, parameter_values=self.parameter_values, experiment=self.experiment, solver=solver)
                    sim.solve(initial_soc=initial_soc)

                
                elif self.last_solution:
                 # Set initial conditions from the last solution
                    new_model=self.model.set_initial_conditions_from(self.last_solution, inplace=False)
                    sim = pybamm.Simulation(new_model, parameter_values=self.parameter_values, experiment=self.experiment, solver=solver)
                    sim.solve()

            
            self.last_solution = sim.solution
            self.results.append(sim)
            self.charging_time += (sim.solution["Time [s]"].data[-1]-sim.solution["Time [s]"].data[0])

        # constant voltage step
            
        final_voltage = round(self.last_solution["Terminal voltage [V]"].data[-1], 3)
        print(f"final voltage: {final_voltage}")
          
        terminations = ["10mA"]

        constant_voltage_step = pybamm.step.voltage(value=final_voltage, termination=terminations)
            

        self.experiment2 = pybamm.Experiment([constant_voltage_step])
        new_model = self.model.set_initial_conditions_from(self.last_solution, inplace=False)
        sim = pybamm.Simulation(new_model, parameter_values=self.parameter_values, experiment=self.experiment2, solver=solver)
        sim.solve()
        self.results.append(sim)
        self.last_solution = sim.solution

        return self.results
    
    def plot(self, solution):
        pybamm.dynamic_plot(solution)

class CustomSimulator:
    def __init__(self):
        self.simulator = PyBaMMSimulator()

    def tem_cutoff(self, variables, cutoff_temperature=313.15):
        """
        Custom temperature cutoff function.
        """
        return cutoff_temperature - variables["X-averaged cell temperature [K]"]

    def create_termination(self, temperature_cutoff=310, voltage_cutoff="4.2V"):
        """
        Creates a temperature-based termination condition.
        """
        tem_termination = pybamm.step.CustomTermination(
            name="Temperature cutoff",
            event_function=lambda variables: self.tem_cutoff(variables, temperature_cutoff)
        )
        return [tem_termination, voltage_cutoff]

    def create_step(self, type, value, termination_conditions, t= np.linspace(0, 10000, 5000), function=None, time_cutoff=20000):
        """
        Create a simulation step.
        
        current: Fixed current value or None if using a custom current function.
        termination_conditions: Termination conditions for the step.
        t: Time range for the custom current function, required if using a custom function.
        custom_current_function: A function that defines custom current, used if 'current' is None.
        """
        if function is not None:
            assert t is not None, "Time range 't' must be provided for custom current functions."
            function_value = np.column_stack([t, function(t)])
            if type == "current":
                return pybamm.step.current(function_value, termination=termination_conditions, duration=time_cutoff)
            if type == "voltage":
                return pybamm.step.voltage(value=function_value, termination=termination_conditions, duration=time_cutoff)
            if type == "power":
                return pybamm.step.power(value=function_value, termination=termination_conditions, duration=time_cutoff)
            if type == "resistance":
                return pybamm.step.resistance(value=function_value, termination=termination_conditions, duration=time_cutoff)

        else:
            if type == "current":
                return pybamm.step.current(value=value, termination=termination_conditions, duration=time_cutoff)
            if type == "voltage":
                return pybamm.step.voltage(value=value, termination=termination_conditions, duration=time_cutoff)
            if type == "power":
                return pybamm.step.power(value=value, termination=termination_conditions, duration=time_cutoff)
            if type == "resistance":
                return pybamm.step.resistance(value=value, termination=termination_conditions, duration=time_cutoff)

    
    def run_simulation_with_steps(self, steps_details):
        """
        Run simulation with predefined steps.
        
        steps_details is a list where each element is a dict with possible keys:
        - 'current': for constant current steps,
        - 'temperature_cutoff', 'voltage_cutoff': for termination conditions,
        - 't', 'custom_current_function': for steps with a custom current profile.
        """
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
            
            step = self.create_step(type = type, value=value, termination_conditions=termination_conditions, t=t, function=function, time_cutoff=time_cutoff)
            steps.append(step)
        
        stages = self.simulator.run_simulation(steps)
        
        return stages
    

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps 

class SimulationResults:
    def __init__(self, stages, initial_soc=0.00001, param=None, stoichiometry=None):
            self.stages = stages
            self.initial_soc = initial_soc
            self.total_simulation_time = 0
            if stoichiometry is not None:
                self.x_0, self.x_100, self.y_0, self.y_100 = stoichiometry
            else:
                DFN_21700 = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
                param = DFN_21700.param
                parameters = pybamm.ParameterValues("Chen2020")
                self.x_0, self.x_100, self.y_100, self.y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(parameters, param)


    def energy_efficiency(self):
        total_energy_in = 0
        total_time_shift = 0

        for sim in self.stages[:-1]:
            solution = sim.solution
            time = solution["Time [s]"].data- solution["Time [s]"].data[0]+ total_time_shift
            total_time_shift = time[-1]
            voltage = solution["Voltage [V]"].entries
            current = solution["Current [A]"].entries
            energy_in = simps(voltage * np.abs(current), time)
            total_energy_in += energy_in

        voltage_out = self.stages[-1].solution["Voltage [V]"].entries
        current_out = self.stages[-1].solution["Current [A]"].entries
        sim = self.stages[-1].solution
        time_out = sim["Time [s]"].data- sim["Time [s]"].data[0]+ total_time_shift
        energy_out = simps(voltage_out* current_out, time_out)
            
        return energy_out/total_energy_in
    
    def estimated_soc(self):
        calculated_soc_at_each_moment = self.calculate_soc_at_each_moment()
        return calculated_soc_at_each_moment[-1][-1]
    
    def calculate_max_temperature(self):
        max_temp = -np.inf  
        for sim in self.stages:
            temp = sim.solution["Cell temperature [K]"].data[-1]
            stage_max_temp = np.max(temp)
            max_temp = max(max_temp, stage_max_temp)
        return max_temp
    
    def total_heat_generated(self):
        total_heat = 0
        for sim in self.stages:
            solution = sim.solution
            total_heating = solution["Total heating [W]"].data
            time = solution["Time [s]"].data
            time_intervals = np.diff(time)

            average_heat_per_interval = 0.5 * (total_heating[:-1] + total_heating[1:])
            total_heat += np.sum(average_heat_per_interval * time_intervals)
        return total_heat

    def total_charging_time(self):
        total_time = 0
        for sim in self.stages:
            solution = sim.solution
            total_time += solution["Time [s]"].data[-1]-solution["Time [s]"].data[0]
        return total_time

    def final_terminal_voltage(self):
        # Assumes the final voltage is from the last stage
        return self.stages[-1].solution["Terminal voltage [V]"].data[-1]

    def termination_reasons(self):
        reasons = []
        for sim in self.stages:
            reasons.append(sim.solution.termination)
        return reasons
    
    def calculate_heat_at_each_moment(self):
        cumulative_heat = 0  # Correctly track heat, not SOC
        heat_time_series = [(0, 0)]  # Initialize with the starting point
        total_time_shift = 0

        for sim in self.stages:
            solution = sim.solution
            heating_data = solution["Total heating [W]"].data
            time_data = solution["Time [s]"].data - solution["Time [s]"].data[0] + total_time_shift
            total_time_shift = time_data[-1]

            for i in range(1, len(time_data)):
                dt = (time_data[i] - time_data[i-1])  # Time difference in seconds
                dW = heating_data[i-1] * dt  # Heat generated over the interval
                cumulative_heat += dW  # Update the cumulative heat

                # Append the time and updated cumulative heat to the series
                heat_time_series.append((time_data[i], cumulative_heat))

        return heat_time_series

    def calculate_soc_at_each_moment(self):
        current_soc = self.initial_soc
        soc_time_series = [(0, self.initial_soc)]  # Initialize SOC time series with initial conditions
        total_time_shift = 0

        for sim in self.stages:
            solution = sim.solution
            # Get the average particle concentrations
            x = solution["Average negative particle concentration"].entries
            y = solution["Average positive particle concentration"].entries
            time_data = solution["Time [s]"].entries - solution["Time [s]"].data[0] + total_time_shift
            total_time_shift = time_data[-1]

            for i in range(len(time_data)):
                cell_SoC_x = 100 * (x[i] - self.x_0) / (self.x_100 - self.x_0)
                cell_SoC_y = 100 * (y[i] - self.y_0) / (self.y_100 - self.y_0)

                current_soc = (cell_SoC_x + cell_SoC_y) / 2
                # Append the time and updated SOC to the series
                soc_time_series.append((time_data[i], current_soc))

        return soc_time_series
    
    def time_to_reach_target_soc(self, target_soc):

        initial_soc = self.initial_soc  # Set the initial SOC for the calculation
        
        # Calculate SOC at each moment throughout the simulation
        soc_time_series = self.calculate_soc_at_each_moment()

        # Iterate through the SOC time series to find when the target SOC is reached
        for time, soc in soc_time_series:
            if (target_soc >= initial_soc and soc >= target_soc) or \
            (target_soc < initial_soc and soc <= target_soc):
                return time
        return None  # Target SOC not reached

    
    def heat_to_reach_target_soc(self, target_soc):

        soc_time_series = self.calculate_soc_at_each_moment()
        target_time = None

        for time_soc, soc in soc_time_series:
            if (target_soc >= self.initial_soc and soc >= target_soc) or \
            (target_soc < self.initial_soc and soc <= target_soc):
                target_time = time_soc
                break

        if target_time is None:
            return None

        heat_time_series = self.calculate_heat_at_each_moment()
        for time_heat, heat in heat_time_series:
            if time_heat == target_time:
                return heat

        return None
    
    def heat_to_reach_target_time(self, target_time):
        total_time_shift = 0
        all_times = np.array([])
        all_heat_generations = np.array([])

        # If target_time is None, set it to a large default value
        if target_time is None:
            target_time = float('inf')

        for sim in self.stages:
            solution = sim.solution

            # Shift the time array for the current stage
            time = solution["Time [s]"].data - solution["Time [s]"].data[0] + total_time_shift
            total_time_shift = time[-1]

            # Concatenate data from each stage
            all_times = np.concatenate([all_times, time])
            all_heat_generations = np.concatenate([all_heat_generations, solution["Total heating [W]"].data])

            # Stop if the total time exceeds the target time
            if total_time_shift > target_time:
                break

        # Trim the data to the target time
        all_times = all_times[all_times < target_time]
        all_heat_generations = all_heat_generations[:len(all_times)]

        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot for heat generation
        plt.plot(all_times[:-1], all_heat_generations[:-1], label='Heat Generation')
        plt.xlabel('Time [s]')
        plt.ylabel('Heat Generation [W]')
        plt.title('Heat Generation Up to Target Time')
        plt.legend()

        plt.show()

        return all_heat_generations[:-1]
        
    def time_to_reach_target_voltage(self, target_voltage):
        accumulated_time = 0
        for sim in self.stages:
            solution = sim.solution
            voltage = solution["Terminal voltage [V]"].data
            time = solution["Time [s]"].data
            for i in range(len(time)):
                if voltage[i] == target_voltage:
                    return accumulated_time + time[i]
            accumulated_time += time[-1]
        return None
    

    def plot_results(self, save_path):
        total_time_shift = 0
        all_times = []
        all_voltages = []
        all_currents = []
        all_heat_generations = []
        all_temperatures = []

        for sim in self.stages:
            solution = sim.solution

            # Shift the time array for the current stage
            time = solution["Time [s]"].data - solution["Time [s]"].data[0] + total_time_shift
            total_time_shift = time[-1]

            # Concatenate data from each stage
            all_times = np.concatenate([all_times, time])
            all_voltages = np.concatenate([all_voltages, solution["Terminal voltage [V]"].data])
            all_currents = np.concatenate([all_currents, solution["Current [A]"].data])
            all_heat_generations = np.concatenate([all_heat_generations, solution["Total heating [W]"].data])
            all_temperatures = np.concatenate([all_temperatures, solution["Cell temperature [K]"].data[-1]])

        # Plotting
        plt.figure(figsize=(12, 24))

        # Plot for temperature
        plt.subplot(5, 1, 1)
        plt.plot(all_times, all_temperatures, label='Temperature')
        plt.ylabel('Temperature [K]')
        plt.title('Temperature Over Stages')
        plt.legend()

        # Plot for Total Heat Generation
        plt.subplot(5, 1, 2)
        plt.plot(all_times, all_heat_generations, label='Heat Generation')
        plt.ylabel('Total Heat Generation [W]')
        plt.title('Heat Generation Over Stages')
        plt.legend()

        # Plot for Terminal Voltage
        plt.subplot(5, 1, 3)
        plt.plot(all_times, all_voltages, label='Voltage')
        plt.ylabel('Terminal Voltage [V]')
        plt.title('Voltage Over Stages')
        plt.legend()

        # Plot for Current
        plt.subplot(5, 1, 4)
        plt.plot(all_times, all_currents, label='Current')
        plt.xlabel('Time [s]')
        plt.ylabel('Current [A]')
        plt.title('Current Over Stages')
        plt.legend()

        # Plot for SOC
        soc_time_series = self.calculate_soc_at_each_moment()
        soc_times, soc_values = zip(*soc_time_series)
        plt.subplot(5, 1, 5)
        plt.plot(soc_times, soc_values, label='State of Charge')
        plt.xlabel('Time [s]')
        plt.ylabel('SOC [%]')
        plt.title('State of Charge Over Time')
        plt.legend()

        # Save the plot if a save path is provided
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")

        # Display the plot
        plt.show()

        # Clear the figure after saving and showing
        plt.close()  # or plt.close() to fully close the figure

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
        print(f"Energy efficiency: {energy_efficiency:.2f}")
        print(f"Max temperature: {self.calculate_max_temperature()} K")

        # Call the plot function to visualize the results
        self.plot_results(save_path)



# Simulation Example

def __main__():

        # Example usage
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import pybamm

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(1, 2, bias=False)
            self.fc2 = nn.Linear(2, 1, bias=False)
            self.relu = nn.ReLU()  # Using ReLU activation function
            self.batch_norm = nn.BatchNorm1d(2)  # Batch normalization
            
        def forward(self, input):
            x = self.fc1(input)
            x = self.batch_norm(x)  # Batch normalization
            x = self.relu(x)  # Using ReLU activation function
            x = self.fc2(x)
            output = torch.sigmoid(x) * 10  # Scale output to range [0, 10]
            return output

    
    def apply_parameters(model, param_list):
        param_dict = {}
        idx = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            param_dict[name] = torch.tensor(param_list[idx:idx+num_params]).reshape(param.shape)
            idx += num_params

        for name, param in model.named_parameters():
            param.data = param_dict[name]

    optimal_params = [-0.5668992911512561, -0.7295636531890959, -0.3517179844135717, -0.7006502656326337, -0.5553572234968246, -0.2270220377482759, 0.8051969510588097, -0.1001000201775446]
    model = NeuralNetwork()
    apply_parameters(model, optimal_params)

    # input
    t_values = np.linspace(0, 3600, 1000).reshape(-1, 1)
    t_tensor = torch.tensor(t_values, dtype=torch.float32)
    output = model(t_tensor).detach().numpy()

    t_values_update = np.linspace(0, 3600, 1000).reshape(-1, 1)
    output_update = -output
    data = np.column_stack((t_values_update, output_update))

    print(data)

    steps_details = [{'type': "current", 'value': data, 'temperature_cutoff': 315, 'voltage_cutoff': "4.2V", 'duration': 3600}]
    simulator = CustomSimulator()
    stages = simulator.run_simulation_with_steps(steps_details)
    results = SimulationResults(stages)
    save_path = "results.png"
    results.show_results(save_path)

    charging_time = results.time_to_reach_target_soc(80)
    heat_to_reach_target_time = results.heat_to_reach_target_time(charging_time)
    print(f"Heat to reach 4.2V: {heat_to_reach_target_time} J")
        

if __name__ == "__main__":
    __main__()
