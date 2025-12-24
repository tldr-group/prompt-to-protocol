import pybamm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps 
import os

def loss_function(Q, target_heat=0.7):
    """
    Heat-based loss function matching SAABO_constant_heating.py / RunCharging implementation.
    Lower is better.
    """
    if isinstance(Q, (list, np.ndarray)):
        N = len(Q)
        heat_loss = np.mean([(Q[i] - target_heat) ** 2 for i in range(N)]) / N
    else:
        heat_loss = 1e6  # Assign a large number if Q is not iterable
    total_loss = heat_loss * 100000
    return total_loss, heat_loss


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
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        # Display the plot
        plt.show()

        # Clear the figure after saving and showing
        plt.clf()  # or plt.close() to fully close the figure

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
    import numpy as np
    import pybamm

    def current_function():
        t = pybamm.t
        Q = 0.4  # target heat (watts)
        R0 = 0.02  # base ohmic resistance (ohms)

        # Smooth bases
        s1 = pybamm.sin(2 * 3.14 * t / 2600 + 0.3)
        c1 = pybamm.cos(2 * 3.14 * t / 3900 - 0.6)
        s2 = pybamm.sin(2 * 3.14 * t / 5200 + 1.0)
        c2 = pybamm.cos(2 * 3.14 * t / 6700 + 0.8)
        s3 = pybamm.sin(2 * 3.14 * t / 8100 - 1.1)
        c3 = pybamm.cos(2 * 3.14 * t / 9500 + 0.2)

        # Nonlinear modulators
        u1 = pybamm.tanh(0.7 * s1 + 0.4 * c2)
        u2 = pybamm.tanh(0.6 * s2 - 0.5 * c1)
        u3 = pybamm.tanh(0.5 * s3 + 0.6 * c3)
        u4 = pybamm.tanh(0.8 * s1 - 0.3 * c3)

        # Positive, time-varying resistances (diverse constructions)
        R1 = R0 * pybamm.exp(0.22 * s1 + 0.10 * c1 + 0.06 * s2 * c2)
        R2 = R0 * (1.05 + 0.25 * s2 + 0.15 * c2 + 0.10 * u1) ** 2
        R3 = R0 * pybamm.exp(0.18 * c3 - 0.08 * s1 + 0.05 * u2)
        R4 = R0 * (0.95 + 0.20 * c1 - 0.10 * s3 + 0.20 * u3) ** 2
        R5 = R0 * pybamm.exp(0.14 * u4 + 0.07 * s2 - 0.04 * c2)
        R6 = R0 * (1.00 + 0.18 * s1 * c1 + 0.12 * s2 * s3) ** 2

        # Smooth softmax weights over six channels
        gamma = 1.0 + 0.3 * pybamm.sin(2 * 3.14 * t / 10400)
        v1 = pybamm.exp(gamma * s1)
        v2 = pybamm.exp(gamma * c1)
        v3 = pybamm.exp(gamma * s2)
        v4 = pybamm.exp(gamma * c2)
        v5 = pybamm.exp(gamma * s3)
        v6 = pybamm.exp(gamma * c3)
        sumv = v1 + v2 + v3 + v4 + v5 + v6
        w1 = v1 / sumv
        w2 = v2 / sumv
        w3 = v3 / sumv
        w4 = v4 / sumv
        w5 = v5 / sumv
        w6 = v6 / sumv

        # Tri-mean over arithmetic, geometric, and harmonic means (A-G-H) with dynamic barycentric weights
        R_arith = w1 * R1 + w2 * R2 + w3 * R3 + w4 * R4 + w5 * R5 + w6 * R6
        R_harm = 1 / (w1 / R1 + w2 / R2 + w3 / R3 + w4 / R4 + w5 / R5 + w6 / R6)
        R_geom = pybamm.exp(
            w1 * pybamm.log(R1)
            + w2 * pybamm.log(R2)
            + w3 * pybamm.log(R3)
            + w4 * pybamm.log(R4)
            + w5 * pybamm.log(R5)
            + w6 * pybamm.log(R6)
        )

        delta = 0.7
        x1 = pybamm.exp(delta * pybamm.sin(2 * 3.14 * t / 7200 + 0.4))
        x2 = pybamm.exp(delta * pybamm.cos(2 * 3.14 * t / 7800 - 0.9))
        x3 = pybamm.exp(delta * pybamm.sin(2 * 3.14 * t / 8400 + 1.1))
        sumx = x1 + x2 + x3
        a = x1 / sumx
        b = x2 / sumx
        c = x3 / sumx

        R_AHG = (R_arith ** a) * (R_geom ** b) * (R_harm ** c)

        # Conductance-side power-harmonic mean, mapped back to resistance
        G1 = 1 / R1
        G2 = 1 / R2
        G3 = 1 / R3
        G4 = 1 / R4
        G5 = 1 / R5
        G6 = 1 / R6
        r = 0.6 + 0.4 * pybamm.sin(2 * 3.14 * t / 8800 - 0.2)  # r in [0.2, 1.0]
        denom = (
            w1 / (G1 ** r)
            + w2 / (G2 ** r)
            + w3 / (G3 ** r)
            + w4 / (G4 ** r)
            + w5 / (G5 ** r)
            + w6 / (G6 ** r)
        )
        G_ph = (1 / denom) ** (1 / r)
        R_condpath = 1 / G_ph

        # Harmonic blending between A-G-H path and conductance path
        alpha = 0.5 + 0.5 * pybamm.sin(2 * 3.14 * t / 7600 + 0.3)
        R_tilde = 1 / (alpha / R_AHG + (1 - alpha) / R_condpath)

        # Smooth envelope to diversify dynamics
        env = pybamm.exp(0.12 * pybamm.tanh(0.5 * s1 - 0.4 * c3 + 0.3 * s2 * c1))
        R_t = R_tilde * env

        # Current targeting constant heat: i^2 * R_t ≈ Q, saturated to [-5, 0] A
        i_mag = pybamm.sqrt(Q / R_t)
        i = -5 * pybamm.tanh(i_mag / 5)
        return i

    # Use PyBaMMSimulator dict format to pass current function
    current_func = current_function()
    t_values = np.linspace(0, 3600, 1000)
    
    print(f"Current function: {current_func}")

    # Use PyBaMMSimulator directly instead of CustomSimulator
    simulator = PyBaMMSimulator()
    
    steps = [{'current_function': current_func, 'time_evals': t_values, 'termination_voltage': 4.2}]
    stages = simulator.run_simulation(steps)
    results = SimulationResults(stages)
    save_path = "results.png"
    results.show_results(save_path)

    charging_time = results.time_to_reach_target_soc(80)
    heat_to_reach_target_time = results.heat_to_reach_target_time(charging_time)
    print(f"Heat to reach 80% SOC: {heat_to_reach_target_time} J")
    total_loss, heat_loss = loss_function(heat_to_reach_target_time, target_heat=0.7)
    print(f"Total loss: {total_loss}, Heat loss: {heat_loss}")
        

if __name__ == "__main__":
    __main__()
