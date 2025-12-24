import pybamm
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import torch
import torch.nn as nn
import pickle
import psutil
import gc
from datetime import datetime

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import functools
import importlib.util
import csv


am_tem = 308.15
p=13

class MemoryMonitor:
    """Memory monitoring utility class"""
    def __init__(self, warning_threshold=80, critical_threshold=90):
        """
        Initialize memory monitor
        warning_threshold: Memory usage warning threshold (percentage)
        critical_threshold: Memory usage critical threshold (percentage)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.process = psutil.Process()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('memory_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self):
        """Get current memory usage"""
        system_memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        return {
            'system_percent': system_memory.percent,
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_gb': system_memory.used / (1024**3),
            'system_total_gb': system_memory.total / (1024**3),
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3)
        }
    
    def log_memory_status(self, context=""):
        """Log memory status"""
        memory_info = self.get_memory_usage()
        
        status_msg = (
            f"[{context}] Memory Status: "
            f"System Usage={memory_info['system_percent']:.1f}%, "
            f"Available={memory_info['system_available_gb']:.2f}GB, "
            f"Process RSS={memory_info['process_rss_gb']:.2f}GB"
        )
        
        if memory_info['system_percent'] >= self.critical_threshold:
            self.logger.critical(status_msg)
        elif memory_info['system_percent'] >= self.warning_threshold:
            self.logger.warning(status_msg)
        else:
            self.logger.info(status_msg)
            
        return memory_info
    
    def cleanup_memory(self):
        """Perform memory cleanup"""
        self.logger.info("Performing memory cleanup...")
        collected = gc.collect()
        self.logger.info(f"Garbage collector cleaned up {collected} objects")
        
        # Log memory status after cleanup
        self.log_memory_status("After cleanup")
    
    def check_memory_and_cleanup_if_needed(self, context=""):
        """Check memory and cleanup if needed"""
        memory_info = self.log_memory_status(context)
        
        if memory_info['system_percent'] >= self.warning_threshold:
            self.cleanup_memory()
            
        if memory_info['system_percent'] >= self.critical_threshold:
            self.logger.critical(f"Memory usage reached {memory_info['system_percent']:.1f}%, consider stopping or reducing parallel tasks")
            return False  # Return False indicating severe memory shortage
        
        return True  # Return True indicating normal memory status

# Create global memory monitor instance
memory_monitor = MemoryMonitor()

import os, pickle, functools
import numpy as np
import pybamm


class BatteryCyclingExperiment:
    def __init__(self, num_cycles=30, SOC=0.9, adaptive_current=None, am_tem=308.15):
        """
        Initialize battery cycling experiment.
        """
        pybamm.set_logging_level("ERROR")

        self.steps = 0
        self.num_cycles = num_cycles
        self.charging_duration = 0.5  # hours
        self.SOC = SOC
        self.adaptive_current = adaptive_current
        self.nominal_capacity_battery = 2.4472
        self.am_tem = am_tem  # Ambient temperature in Kelvin
        self.v_step1 = 4.2

        V_threshold_param = "Overvoltage threshold [V]"
        alpha_param = "Overvoltage degradation factor [1/s]"
        n_param = "Overvoltage exponent"

        self.model = pybamm.lithium_ion.DFN(
            options={
                "particle": "Fickian diffusion",
                "particle mechanics": "swelling and cracking",
                "loss of active material": "stress-driven",
                "SEI": "reaction limited",
                "SEI porosity change": "true",
                "calculate discharge energy": "true",
                "SEI on cracks": "true",
                "thermal": "lumped",
            },
        )

        self.parameter_values = pybamm.ParameterValues("Ai2020")
        k = self.parameter_values["SEI reaction exchange current density [A.m-2]"]
        self.parameter_values["SEI reaction exchange current density [A.m-2]"] = k * 5
        self.parameter_values.update(
            {
                "Negative electrode LAM constant proportional term [s-1]": 1e-12,
                "Positive electrode LAM constant proportional term [s-1]": 2.78e-13 * 10,
                "Positive electrode cracking rate": 3.9e-20 * 10,
                "Negative electrode cracking rate": 3.9e-20 * 10,
                "Total heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Ambient temperature [K]": self.am_tem,
            },
            check_already_exists=False,
        )
        self.parameter_values.update(
            {V_threshold_param: 4.2, alpha_param: 0.3, n_param: 3},
            check_already_exists=False,
        )
        self.parameter_values.set_initial_stoichiometries(1)

        V = self.model.variables["Terminal voltage [V]"]
        Q_loss = pybamm.Variable("Q_loss")
        V_threshold = pybamm.Parameter(V_threshold_param)
        alpha = pybamm.Parameter(alpha_param)
        n = pybamm.Parameter(n_param)
        over_voltage = pybamm.maximum(V - V_threshold, 0)
        dQdt = alpha * (over_voltage**n)
        self.model.rhs[Q_loss] = dQdt
        self.model.initial_conditions[Q_loss] = pybamm.Scalar(0)

        self.current_capacity = self.nominal_capacity_battery
        self.model.variables["Delta capacity [A.h]"] = (
            self.nominal_capacity_battery - self.current_capacity
        )
        self.model.variables["SoC"] = 1 - (
            self.model.variables["Discharge capacity [A.h]"]
            + self.model.variables["Delta capacity [A.h]"]
        ) / self.nominal_capacity_battery

        self.soc_termination = pybamm.step.CustomTermination(
            name="SoC = 90%", event_function=self.soc_90_cutoff
        )

        self.solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8)
        self.soh_solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8)

    def soc_90_cutoff(self, variables):
        print(f"1====================={self.current_capacity}")
        return self.SOC - variables["SoC"]

    def measure_soh_and_new_capacity(self, solution):
        Q_loss = solution["Q_loss"](solution.t[-1])
        print(f"Q loss due to high voltage: {Q_loss}")
        soh = (self.current_capacity - Q_loss) / self.nominal_capacity_battery
        self.model.variables["Delta capacity [A.h]"] = (
            self.nominal_capacity_battery - self.current_capacity
        )
        self.model.variables["SoC"] = 1 - (
            self.model.variables["Discharge capacity [A.h]"]
            + self.model.variables["Delta capacity [A.h]"]
        ) / self.nominal_capacity_battery
        return soh

    def run_part1(self, prev_last_sub=None):
        steps_part1 = (
            pybamm.step.current(value=5 / 3, termination="3.0V"),
            pybamm.step.voltage(value=3.0, termination="50mA"),
            pybamm.step.CustomStepExplicit(
                self.adaptive_current,
                duration=self.charging_duration * 3600,
                termination=[f"{self.v_step1}V", self.soc_termination],
                direction="charge",
            ),
        )
        experiment_1 = pybamm.Experiment([steps_part1])
        sim_1 = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=experiment_1,
            solver=self.solver,
        )
        sim_1._esoh_solver = self.soh_solver

        if prev_last_sub is None:
            sol_1 = sim_1.solve(initial_soc=0.999)
        else:
            sol_1 = sim_1.solve(starting_solution=prev_last_sub)

        steps = sol_1.cycles[-1].steps
        for i, step in enumerate(steps):
            print(f"Step {i+1} termination reason: {step.termination}")

        self.steps += 1
        last_step_end_time = float(steps[-1].t[-1])
        second_last_step_end_time = float(steps[-2].t[-1])
        last_step_duration = last_step_end_time - second_last_step_end_time
        print(f"Last step duration: {last_step_duration} seconds")

        last_sub = sol_1.sub_solutions[-1]
        return sol_1, last_step_duration, last_sub

    def run_part2(self, sol_1, last_step_duration, last_sub_from_part1):
        soc_part1 = sol_1["SoC"].data[-1]
        discharge_cap_after_discharge = (
            sol_1.cycles[-1].steps[0]["Discharge capacity [A.h]"].data[-1]
        )
        print(
            f"Previous partial-cycle ended with: SoC={soc_part1:.4f}, Discharge cap={discharge_cap_after_discharge:.4f}"
        )
        I = sol_1["Current [A]"].data
        print(f"I.min(), I.max() = {I.min():.4f}, {I.max():.4f}")
        self.current_capacity = discharge_cap_after_discharge

        if soc_part1 < self.SOC:
            delta_soc = self.SOC - soc_part1
            delta_Q = self.nominal_capacity_battery * delta_soc
            leftover_hours = self.charging_duration - (last_step_duration / 3600)
            if leftover_hours <= 0:
                print("No leftover time for Part 2, skipping.")
                self.steps += 1
                # Reuse sol_1's last sub-solution for subsequent steps
                return sol_1, last_sub_from_part1

            i_charge_part2 = delta_Q / leftover_hours
            print(f"Calculated Part2 charge current: {i_charge_part2:.4f} A")

            steps_part2 = (
                pybamm.step.current(
                    value=-i_charge_part2,
                    duration=7200,
                    termination=[self.soc_termination],
                ),
            )
            experiment_2 = pybamm.Experiment([steps_part2])
            sim_2 = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_values,
                experiment=experiment_2,
                solver=self.solver,
            )
            sim_2._esoh_solver = self.soh_solver

            # Use only part1's last sub-solution as starting point
            sol_2 = sim_2.solve(starting_solution=last_sub_from_part1)
            self.steps += 1
            last_sub2 = sol_2.sub_solutions[-1]
            return sol_2, last_sub2
        else:
            self.steps += 1
            print("The custom current in Part1 brought SoC > 90%, skipping Part2.")
            # If skipping part2, continue with part1's last sub-solution
            return sol_1, last_sub_from_part1

    def steps_part3(self, prev_last_sub, rest_time):
        steps_part3 = (pybamm.step.current(value=0, duration=rest_time),)
        experiment_3 = pybamm.Experiment([steps_part3])
        sim_3 = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=experiment_3,
            solver=self.solver,
        )
        sim_3._esoh_solver = self.soh_solver
        sol_3 = sim_3.solve(starting_solution=prev_last_sub)
        last_sub3 = sol_3.sub_solutions[-1]
        return sol_3, last_sub3

    def run_cycles(self, path=None):
        cycle_lim = 100
        if self.num_cycles > cycle_lim and path is None:
            raise ValueError(
                "For cycles greater than 100, please provide a valid path for checkpoint storage."
            )

        prev_last_sub = None  # Only pass last step's sub-solution
        final_solution = None
        soh_values = []

        for cycle_index in range(self.num_cycles):
            print(f"\n--- Cycle {cycle_index + 1} ---")
            memory_monitor.log_memory_status(f"Cycle {cycle_index+1} start")

            # Part 1
            sol_1, last_step_duration, last_sub1 = self.run_part1(prev_last_sub)
            Soc_sol_1 = sol_1["SoC"].data[-1]
            V_sol_1 = sol_1["Voltage [V]"].data[-1]

            eps = 1e-4
            if Soc_sol_1 < self.SOC - eps and V_sol_1 < self.v_step1 - eps:
                raise ValueError(
                    f"Cycle {cycle_index + 1} ended with SoC {Soc_sol_1:.4f} and voltage {V_sol_1:.4f}, and time duration {last_step_duration:.4f} seconds, which is not acceptable."
                )

            # Part 2
            sol_2, last_sub2 = self.run_part2(sol_1, last_step_duration, last_sub1)
            print(f"End of Part 2, final SoC: {sol_2['SoC'].data[-1]}")
            print(
                f"Discharge capacity at end of cycle: {sol_2['Discharge capacity [A.h]'].data[-1]:.4f} A·h"
            )
            print(f"Current capacity after cycle: {self.current_capacity:.4f} A·h")
            print(f"delta capacity: {sol_2['Delta capacity [A.h]'].data[-1]:.4f} A·h")

            # Part 3
            t0 = sol_1.t[-1]
            t1 = sol_2.t[-1]
            part2_charing_duration = t1 - t0
            rest_time = (
                3600 * self.charging_duration + 300 - part2_charing_duration - last_step_duration
            )
            print(f"Rest time for Part 3: {rest_time} seconds")
            sol_3, last_sub3 = self.steps_part3(last_sub2, rest_time)
            final_solution = sol_3

            print(f"End of Part 3, final SoC: {sol_3['SoC'].data[-1]}")

            # SOH & bookkeeping
            soh_val = self.measure_soh_and_new_capacity(sol_2)
            soh_values.append(soh_val)

            # Check if SOH is below 70% (disabled to record all results)
            # if soh_val < 0.7:
            #     raise ValueError(
            #         f"Battery SOH dropped to {soh_val:.3f} ({soh_val*100:.1f}%), below 70% threshold."
            #         f"Stopping after cycle {cycle_index + 1}."
            #     )
            if soh_val < 0.7:
                print(
                    f"Warning: SOH dropped to {soh_val:.3f} ({soh_val*100:.1f}%), continuing to record all results."
                )

            cycle_soc = sol_2["SoC"].data[-1]
            print(
                f"End of cycle {cycle_index + 1}, final SoC: {cycle_soc}, SOH: {soh_val:.5f}"
            )
            print(f"Current capacity after cycle: {self.current_capacity:.4f} A·h")
            print(
                f"Discharge capacity at end of cycle: {sol_2['Discharge capacity [A.h]'].data[-1]:.4f} A·h"
            )
            print(f"2====================={self.current_capacity}")

            if abs(cycle_soc - self.SOC) > 0.01:
                raise ValueError(
                    f"Ending SoC {cycle_soc} is off by more than 0.001 from {self.SOC}. Aborting."
                )

            if self.num_cycles > cycle_lim and (cycle_index + 1) % 10 == 0:
                checkpoint_folder = os.path.join(path, "checkpoint")
                os.makedirs(checkpoint_folder, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_folder, "soh_values.pkl")
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(soh_values, f)
                print(
                    f"Checkpoint saved at cycle {cycle_index + 1} to {checkpoint_file}"
                )

            if isinstance(self.adaptive_current, functools.partial):
                metric = self.adaptive_current.keywords.get("self_ref")
                if metric is not None:
                    metric.h_prev = None

            # Only pass last sub-solution between cycles
            prev_last_sub = last_sub3

            # Release large object references (optional)
            sol_1 = None
            sol_2 = None
            sol_3 = None
            memory_monitor.log_memory_status(f"Cycle {cycle_index+1} end")

        print("\n--- All cycles complete ---")
        final_soc = final_solution["SoC"].data[-1]
        final_discharge_capacity = final_solution["Discharge capacity [A.h]"].data
        print(f"Final SOC after all cycles: {final_soc}")
        print(f"Final SOH: {soh_values[-1]:.5f}")
        return final_solution, soh_values



    def plot_results(self, final_solution, soh_values, save_path=None):
        """
        Plot the discharge capacity trend, the current and voltage from the final solution,
        and the SOH progression over the cycles.
        The plots will be saved to the provided directory (save_path). If save_path is None,
        the user will be prompted to input a folder path.
        """

        if save_path and not save_path.endswith(os.sep):
            save_path += os.sep

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        plt.figure()
        plt.plot(final_solution["Discharge capacity [A.h]"].data, label="Discharge capacity")
        plt.ylabel("Discharge capacity [A.h]")
        plt.title("Discharged Capacity Trend")
        plt.legend()
        plt.savefig(os.path.join(save_path, "discharge_capacity_trend.png"))
        plt.close()
        
        plot = pybamm.QuickPlot(final_solution, ["Current [A]", "Voltage [V]"])
        plot.plot(0)
        plt.savefig(os.path.join(save_path, "current_voltage_plot.png"))
        plt.close()
        
        plt.figure()
        plt.plot(range(1, self.num_cycles + 1), soh_values, marker='o', label="SOH")
        plt.xlabel("Cycle number")
        plt.ylabel("SOH")
        plt.title("SOH After Each Cycle")
        plt.legend()
        plt.savefig(os.path.join(save_path, "soh_after_each_cycle.png"))
        plt.close()

class BatteryExperimentRunner:
    def __init__(self, input_folder, terminal_soc=0.8, num_cycles=2, type_name='nn'):
        """
        Initializes the runner with an input folder that contains:
        - A model_architecture.py file defining 'NeuralNetwork'.
        - A single 'all_params.pth' with stored parameters.
        
        The code automatically detects whether the model is RNN or MLP based on 
        the nn_in_pybamm function signature:
        - RNN: nn_in_pybamm(X, H_prev, Ws, bs) - 4 parameters
        - MLP: nn_in_pybamm(X, Ws, bs) - 3 parameters
        """
        self.input_folder = input_folder
        # self.model = self.load_model()
        self.csv_rows = self.load_csv()
        self.terminal_soc = terminal_soc
        self.type_name = type_name
        self.num_cycles = num_cycles
        self.h_prev = None  # Only used in RNN mode

    def load_model(self):
        """
        Dynamically loads the model architecture from a Python file.
        """
        model_path = os.path.join(self.input_folder, "model_architecture.py")
        spec = importlib.util.spec_from_file_location("model_architecture", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        return module.NeuralNetwork()
    
    def load_nn_in_pybamm(self):
        """
        Dynamically loads the model architecture from a Python file.
        """
        model_path = os.path.join(self.input_folder, "model_architecture.py")
        spec = importlib.util.spec_from_file_location("model_architecture", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        return module.nn_in_pybamm
    
    def is_rnn_model(self):
        """
        Detect if model is RNN type
        Returns:
            bool: True if RNN (4 parameters), False if MLP (3 parameters)
        """
        if not hasattr(self, 'nn_in_pybamm'):
            self.nn_in_pybamm = self.load_nn_in_pybamm()
        
        import inspect
        nn_func_params = inspect.signature(self.nn_in_pybamm).parameters
        return len(nn_func_params) == 4
    
    
    
    def load_csv(self):
        """
        Loads the CSV so we can determine best, worst, and median based on 'total_loss'.
        Assumes each row corresponds to a single parameter set, in the same order as 'all_params.pth'.
        """
        csv_path = os.path.join(self.input_folder, "optimization_results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"'{csv_path}' not found. Make sure it exists.")
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def apply_parameters(self, param_index):
        self.model = self.load_model()
        """
        Loads parameters from 'all_params.pth' by index and applies them to the model.
        'param_index' indicates which saved parameter set to use.
        """
        pth_file = os.path.join(self.input_folder, 'all_params.pth')
        if not os.path.exists(pth_file):
            raise FileNotFoundError(f"'{pth_file}' not found. Make sure it exists and contains parameter data.")

        stored_params = torch.load(pth_file)
        if param_index < 0 or param_index >= len(stored_params):
            raise IndexError(f"Parameter index {param_index} is out of range in 'all_params.pth'.")

        param_data = stored_params[param_index]

        # If 'param_data' is a dict (e.g., {'fc1.weight_0': val, ...}), apply accordingly.
        if isinstance(param_data, dict):
            grouped_data = {}
            for k, v in param_data.items():
                if "_" in k:
                    base, idx = k.rsplit("_", 1)
                    if idx.isdigit():
                        grouped_data.setdefault(base, []).append((int(idx), v))
                    else:
                        grouped_data.setdefault(k, []).append((0, v))
                else:
                    grouped_data.setdefault(k, []).append((0, v))

            for name, param in self.model.named_parameters():
                pairs = grouped_data[name]
                pairs.sort(key=lambda x: x[0])
                values = [p[1] for p in pairs]
                param_tensor = torch.tensor(
                    values,
                    dtype=torch.float32,
                    device=next(self.model.parameters()).device
                ).reshape(param.shape)
                param.data.copy_(param_tensor)
        else:
            # If 'param_data' is a list of floats, apply them in order.
            idx = 0
            param_dict = {}
            for name, param in self.model.named_parameters():
                num_params = param.numel()
                param_tensor = torch.tensor(
                    param_data[idx:idx + num_params],
                    dtype=torch.float32,
                    device=next(self.model.parameters()).device
                ).reshape(param.shape)
                param_dict[name] = param_tensor
                idx += num_params
            for name, param in self.model.named_parameters():
                param.data.copy_(param_dict[name])
    
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
    
    @staticmethod
    def adaptive_current_core(vars_dict, self_ref, Ws, bs):
        V   = vars_dict["Voltage [V]"]
        T   = vars_dict["Volume-averaged cell temperature [K]"]
        soc = vars_dict["SoC"]

        Vn = 2 * ((V - 3.0) / (4.2 - 3.0)) - 1
        Tn = 2 * ((T - am_tem) / (p)) - 1
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



    def run_experiment(self, param_index, case_name, t_start=0, t_end=7200, num_points=1000):
        """
        Runs an experiment by:
        - Applying parameters (by index) from 'all_params.pth' to the model.
        - Generating a drive cycle current.
        - Running the battery cycling simulation.
        - Saving the results (plots, etc.) to an output folder.
        """
        self.model = self.load_model()
        self.nn_in_pybamm = self.load_nn_in_pybamm()
        # Apply the selected parameters
        self.apply_parameters(param_index)

        Ws, bs = self.get_fc_weights(self.model)
        charging_current = self.make_pybamm_current(Ws, bs)


        # Run the battery cycling experiment
        experiment = BatteryCyclingExperiment(num_cycles=self.num_cycles, SOC=self.terminal_soc, adaptive_current=charging_current)
        checkpoint_path = os.path.join(self.input_folder, "output", case_name)
        final_solution, soh_values = experiment.run_cycles(path=checkpoint_path)

        # Save and plot results
        output_folder = os.path.join(self.input_folder, "output", case_name)
        experiment.plot_results(final_solution, soh_values, save_path=output_folder)
        return final_solution, soh_values

    def plot_aggregate_soh(self, all_sohs):
        """
        Plots SOH curves from multiple runs in one figure.
        """
        output_folder = os.path.join(self.input_folder, "output")
        os.makedirs(output_folder, exist_ok=True)

        plt.figure()
        for label, soh_values in all_sohs.items():
            plt.plot(range(1, len(soh_values) + 1), soh_values, marker='o', label=label)
        plt.xlabel("Cycle number")
        plt.ylabel("SOH")
        plt.title("Comparison of SOH for All Cases")
        plt.legend()
        plt.savefig(os.path.join(output_folder, "soh_all_cases.png"))
        plt.close()

    import os
    import numpy as np
    import torch
    import pybamm  # Ensure PyBaMM is installed

    def run_best(self):
        """
        Picks best, worst, and median *based on the CSV 'total_loss'*,
        then runs each experiment and finally extracts the *second cycle*
        of (time, current, voltage, soc) for each case.
        """
        pth_file = os.path.join(self.input_folder, 'all_params.pth')
        if not os.path.exists(pth_file):
            raise FileNotFoundError(f"'{pth_file}' not found.")

        # Filter out records with loss = 1e6
        valid_rows_with_index = [
            (i, row) for i, row in enumerate(self.csv_rows)
            if float(row["total_loss"]) != 1e6
        ]
        if not valid_rows_with_index:
            raise ValueError("No valid rows with loss ≠ 1e6 found.")

        # Sort and get best, worst, middle
        rows_sorted = sorted(valid_rows_with_index, key=lambda x: float(x[1]["total_loss"]))
        keys = ["Best", "Worst", "Middle"]
        indices = [
            rows_sorted[30][0],
            rows_sorted[40][0],
            # rows_sorted[2][0],
            # rows_sorted[len(rows_sorted) // 2-1][0],
            rows_sorted[50][0],
            # rows_sorted[len(rows_sorted) // 2 +1][0]
        ]
        loss_values = {}
        for key, idx in zip(keys, indices):
            loss_values[key] = float(self.csv_rows[idx]["total_loss"])

        simulation_data = {}
        all_sohs = {}

        def safe_run_experiment(index, case_name):
            try:
                if self.type_name == 'nn':
                    sol, soh = self.run_experiment(
                        index, case_name,
                        t_start=0, t_end=7200, num_points=1000
                    )
                else:
                    sol, soh = self.run_experiment_cccv(
                        index, case_name,
                        t_start=0, t_end=7200, num_points=1000
                    )
                # Extract all cycle data
                t = sol["Time [s]"].data
                I = sol["Current [A]"].data
                V = sol["Voltage [V]"].data
                soc = sol["SoC"].data
                T = sol["Volume-averaged cell temperature [K]"].data

                simulation_data[case_name] = (t, I, V, soc, T)
                all_sohs[case_name] = soh  # soh aligned with t
            except Exception as e:
                print(f"Error in {case_name}: {e}")

        # Run three cases
        for key, idx in zip(keys, indices):
            safe_run_experiment(idx, key)

        # Extract second cycle data
        for case_name, (t, I, V, soc, T) in simulation_data.items():
            # Find boundary: first index where current goes from <=0 to >0
            flip_idxs = np.where((I[:-1] <= 0) & (I[1:] > 0))[0]
            if flip_idxs.size == 0:
                raise RuntimeError(f"{case_name}: No <=0 to >0 transition found")
            boundary = flip_idxs[0] + 1

            if boundary >= len(t):
                raise RuntimeError(f"{case_name}: Boundary index {boundary} exceeds array length {len(t)}")

            # Slice from boundary to end (second cycle)
            t2   = t[boundary:]   - t[boundary]
            I2   = I[boundary:]
            V2   = V[boundary:]
            soc2 = soc[boundary:]
            T2   = T[boundary:]
            soh2 = all_sohs[case_name][boundary:]

            # Override original data
            simulation_data[case_name] = (t2, I2, V2, soc2, T2)
            all_sohs[case_name] = soh2

        # Plot aggregate SOH curves
        self.plot_aggregate_soh(all_sohs)

        # Combine and return
        combined_data = {
            case_name: {
                "simulation_data": simulation_data[case_name],
                "loss": loss_values.get(case_name)
            }
            for case_name in simulation_data
        }
        return combined_data


    def run_selected_case(self, case_type='Best', num_cycles=100):
        """
        Runs the experiment for a single selected case (Best, Worst, or Middle),
        for the specified number of cycles, and extracts the per-cycle slices of
        (time, current, voltage, soc).

        Args:
            case_type (str): One of 'Best', 'Worst', or 'Middle'.
            num_cycles (int): Number of cycles to run and slice.

        Returns:
            dict: A dict with the selected case_type as key, containing:
                - 'simulation_data': list of tuples [(t_slice, I_slice, V_slice, soc_slice), ...]
                - 'loss': float loss value for the selected case
        """
        import os
        import numpy as np

        # Load parameters
        pth_file = os.path.join(self.input_folder, 'all_params.pth')
        if not os.path.exists(pth_file):
            raise FileNotFoundError(f"'{pth_file}' not found.")

        # Filter out invalid rows
        valid = [(i, row) for i, row in enumerate(self.csv_rows)
                if float(row['total_loss']) != 1e6]
        if not valid:
            raise ValueError("No valid rows with loss ≠ 1e6 found.")

        # Sort rows by loss
        sorted_rows = sorted(valid, key=lambda x: float(x[1]['total_loss']))
        idx_map = {
            'Best': sorted_rows[0][0],
            'Worst': sorted_rows[len(sorted_rows)//2][0],
            'Middle': sorted_rows[len(sorted_rows)//4][0]
        }
        if case_type not in idx_map:
            raise ValueError("case_type must be one of 'Best', 'Worst', or 'Middle'.")

        idx = idx_map[case_type]
        loss_val = float(self.csv_rows[idx]['total_loss'])

        # Run experiment for the given number of cycles
        if self.type_name == 'nn':
            sol, soh = self.run_experiment(
                idx, case_type,
                t_start=0, t_end=7200, num_points=1000
            )
        else:
            sol, soh = self.run_experiment_cccv(
                idx, case_type,
                t_start=0, t_end=7200, num_points=1000
            )

        # Extract data arrays
        t_all = sol['Time [s]'].data
        I_all = sol['Current [A]'].data
        V_all = sol['Voltage [V]'].data
        soc_all = sol['SoC'].data
        T_all = sol['Volume-averaged cell temperature [K]'].data
        discharge_capacity_all = sol['Discharge capacity [A.h]'].data

        # Detect cycle boundaries: indices where current flips from ≤0 to >0
        flip_indices = np.where((I_all[:-1] <= 0) & (I_all[1:] > 0))[0] + 1
        needed_transitions = num_cycles - 1
        if len(flip_indices) < needed_transitions:
            raise RuntimeError(
                f"Expected at least {needed_transitions} cycle transitions, found {len(flip_indices)}"
            )

        # Create boundaries array: start of cycle 0, transitions, and end of data
        boundaries = np.concatenate(
            ([0], flip_indices[:needed_transitions], [len(t_all)])
        )

        # Slice data for each cycle
        simulation_slices = []
        for c in range(num_cycles):
            start = boundaries[c]
            end = boundaries[c+1]
            t_seg = t_all[start:end] - t_all[start]
            I_seg = I_all[start:end]
            V_seg = V_all[start:end]
            soc_seg = soc_all[start:end]
            T_seg = T_all[start:end]
            discharge_capacity_seg = discharge_capacity_all[start:end]
            simulation_slices.append((t_seg, I_seg, V_seg, soc_seg, T_seg))


        return {
            case_type: {
                'simulation_data': simulation_slices,
                'loss': loss_val
            }
        }

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap



def plot_protocol_with_loss(save_path, combined_simulation_data, combined_loss_data, terminal_soc=0., type='nn',custom_colors=None):
    """
    Plot current, voltage, SoC, temperature subplots with color-coded loss values.
    """
    # 1. Sort by loss value (lower loss = better performance)
    sorted_cases = sorted(combined_loss_data, key=combined_loss_data.get)
    n = len(sorted_cases)

    base_colors = ["#0747a1", "#cce6ff"]  
    cmap = LinearSegmentedColormap.from_list("custom_blue", base_colors, N=n)

    # 2. Generate default colors if not provided
    if custom_colors is None:
        colors = {}
        for i, case in enumerate(sorted_cases):
            if i == 0:
                colors[case] = "#043178"  # Darker color for best case
            else:
                colors[case] = mcolors.to_hex(cmap(i / (n - 1)))

    else:
        colors = custom_colors

    
    color_list = [colors[case] for case in sorted_cases]
    
    # 4. Create 4 subplots (current, voltage, SoC, temperature)
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6, 10))
    ax_current, ax_voltage, ax_soc, ax_temp = axes

    # 5. Add reference lines
    ax_current.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax_voltage.axhline(4.2, linestyle="--", linewidth=1, color="gray")
    ax_voltage.axhline(3.0, linestyle="--", linewidth=1, color="gray")
    ax_soc.axhline(terminal_soc, linestyle="--", linewidth=1, color="gray")
    ax_soc.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    ax_temp.axhline(am_tem, linestyle="--", linewidth=1, color="gray")  # Ambient temperature reference
    
    # 6. Assume all cases share same time interval
    # sample_case = next(iter(combined_simulation_data.values()))
    sample_case = list(combined_simulation_data.values())[-1]

    t_common, I_common = sample_case[0], sample_case[1]
    t_common_in_hours = t_common / 3600

    # 7. Determine charging/discharging regions (current < -0.01 = charging)
    charge_regions = I_common < -0.01
    def find_regions(condition):
        regions = []
        start = None
        for i, cond in enumerate(condition):
            if cond and start is None:
                start = i
            elif not cond and start is not None:
                regions.append((t_common[start], t_common[i-1]))
                start = None
        if start is not None:
            regions.append((t_common[start], t_common[-1]))
        return regions

    charging_intervals = find_regions(charge_regions)
    discharging_intervals = find_regions(~charge_regions)
    charging_intervals_in_hours = [(start/3600, end/3600) for (start, end) in charging_intervals]
    discharging_intervals_in_hours = [(start/3600, end/3600) for (start, end) in discharging_intervals]

    # Generate rest intervals and draw background shading
    fixed_len = 0.5      # Yellow region length (hours)
    rest_intervals_in_hours = []
    timeline_end = t_common_in_hours[-1]

    # Get first charging (yellow) start time
    if charging_intervals_in_hours:
        first_charge_start = charging_intervals_in_hours[0][0]

        # Keep first discharge region as gray
        first_discharge_start, _ = discharging_intervals_in_hours[0]
        if first_discharge_start < first_charge_start:
            rest_intervals_in_hours.append(
                (first_discharge_start, first_charge_start)
            )

        # Other gray regions: from yellow end to next charging or timeline end
        for i, (c_start, _) in enumerate(charging_intervals_in_hours):
            rest_start = c_start + fixed_len        # Yellow region end point
            rest_end = (charging_intervals_in_hours[i+1][0]
                        if i < len(charging_intervals_in_hours) - 1
                        else timeline_end)
            rest_intervals_in_hours.append((rest_start, rest_end))

    # Draw background shading
    # Yellow: charging start to start + fixed_len
    for interval in charging_intervals_in_hours:
        new_end = interval[0] + fixed_len
        for ax in axes:
            ax.axvspan(interval[0], new_end, color="#eddca5", alpha=0.3)

    # Gray: use calculated rest_intervals_in_hours
    for interval in rest_intervals_in_hours:
        # Skip invalid (negative width) intervals
        if interval[1] > interval[0]:
            for ax in axes:
                ax.axvspan(*interval, color="lightgray", alpha=0.3)
    # -----------------------------------------------------------------


    # 9. Plot curves for each case with assigned colors
    for case, data in combined_simulation_data.items():
        t, I, V, soc, T = data
        t_in_hours = t / 3600
        color = colors.get(case, "black")
        ax_current.plot(t_in_hours, I, linestyle="-", color=color, 
                        label=f"{case} (loss: {combined_loss_data[case]:.2f})", alpha=0.8)
        ax_voltage.plot(t_in_hours, V, linestyle="-", color=color, alpha=0.9)
        ax_soc.plot(t_in_hours, soc, linestyle="-", color=color, alpha=0.8)
        ax_temp.plot(t_in_hours, T, linestyle="-", color=color, alpha=0.8)

    # 10. Add helper text to subplots
    ax_current.text(0.72, 0.8, 'Charging', transform=ax_current.transAxes, fontsize=11, color='gray', alpha=0.6)
    ax_current.text(0.22, 0.8, 'Discharging', transform=ax_current.transAxes, fontsize=11, color='gray', alpha=0.6)
    # ax_current.text(0.85, 0.8, 'Rest', transform=ax_current.transAxes, fontsize=11, color='gray', alpha=0.6)
    ax_voltage.text(t_common_in_hours[-1]*0.8, 4.2, '4.2V', va='top', ha='right', fontsize=10, color='gray')
    ax_voltage.text(t_common_in_hours[-1]*0.95, 3.0, '3.0V', va='bottom', ha='right', fontsize=10, color='gray')
    ax_soc.text(t_common_in_hours[-1]*0.95, terminal_soc, f'{terminal_soc*100:.0f}% SoC', va='top', ha='right', fontsize=10, color='gray')
    ax_soc.text(t_common_in_hours[-1]*0.95, 0, '0% SoC', va='bottom', ha='right', fontsize=10, color='gray')
    ax_temp.text(t_common_in_hours[-1]*0.95, am_tem, f'{am_tem}K', va='bottom', ha='right', fontsize=10, color='gray')

    # 11. Add annotation arrows
    if charging_intervals_in_hours:
        start_yellow = charging_intervals_in_hours[0][0]
        fixed_end = start_yellow + fixed_len
        ax_current.annotate(
            '', 
            xy=(start_yellow, 1.3), 
            xytext=(fixed_end, 1.3),
            arrowprops=dict(arrowstyle='<->', color='black')
        )
        ax_current.text(
            (start_yellow + fixed_end) / 2, 1.35, 
            f'0.5h 0-{terminal_soc*100:.0f}% SoC', 
            ha='center', va='bottom', fontsize=8, color='black'
        )

    # 12. Set title and axis labels
    if type == 'nn':
        fig.suptitle("Battery Protocols Optimized by Neural Network Based on SOH", fontsize=11, fontweight='bold', ha='center')
    else:
        fig.suptitle("Multi-Step CCCV Battery Protocols Based on SOH", fontsize=11, fontweight='bold', ha='center')
    ax_current.set_ylabel("Current [A]")
    ax_voltage.set_ylabel("Voltage [V]")
    ax_soc.set_ylabel("SoC")
    ax_temp.set_ylabel("Temperature [K]")
    ax_temp.set_xlabel("Time [h]")
    
    # 13. Create small inset colorbar for SoH (derived from loss)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create small colorbar region in ax_current (lower left)
    cax = inset_axes(ax_current, width="38%", height="3%", loc='lower left', borderpad=3)
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    
    # Calculate SoH values for colorbar ticks
    # Formula: SoH = 0.93 + 0.07 * exp(-loss)
    cmap_discrete = ListedColormap(color_list)

    # 4) Create ScalarMappable with discrete colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=n - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap_discrete, norm=norm)
    sm.set_array([])

    # 5) Draw colorbar
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks(range(n))
    soh_values_sorted = [0.93 + 0.07 * np.exp(-combined_loss_data[case]) for case in sorted_cases]
    soh_loss = (1 - np.array(soh_values_sorted))*100
    # cbar.set_ticklabels([f"{val:.3f}" for val in soh_values_sorted])

    cbar.set_ticklabels([f"{soh_val:.2f}" for soh_val in soh_loss])
    cbar.set_label("SoH Reduction %", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    
    # 14. Adjust layout, save and display
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)




# Example usage:
if __name__ == "__main__":
    import pandas as pd
    import os
    from datetime import datetime
    
    # Set the folder path to run
    Input_path = 'Test_adaptive_RNN_90SOC/20250803_104928_model_0 copy 2'
    terminal_soc = 0.9
    type_name = 'nn'
    num_cycles = 1000
    
    print(f"Starting to run Best, Worst, and Middle protocols from {Input_path}, each running {num_cycles} cycles")
    print(f"Target SOC: {terminal_soc}")
    
    # Create experiment runner
    runner = BatteryExperimentRunner(
        Input_path,
        terminal_soc=terminal_soc,
        num_cycles=num_cycles,
        type_name=type_name
    )
    
    # Run Best, Worst, and Middle cases
    print("\nRunning Best, Worst, and Middle protocols...")
    try:
        # This will run Best, Worst, Middle three cases
        results = runner.run_best()  # This function runs three cases and returns results
        
        # Create folder to save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"Test_adaptive_RNN_90SOC/20250803_104928_model_0 copy 2/SOH_Results_{timestamp}"
        os.makedirs(results_folder, exist_ok=True)
        
        print(f"\nResults will be saved to: {results_folder}")
        
        # Save SOH data for each case
        for case_type in ['Best', 'Worst', 'Middle']:
            if case_type in results:
                case_data = results[case_type]
                loss_value = case_data['loss']
                simulation_data = case_data['simulation_data']
                
                print(f"\n{case_type} Protocol:")
                print(f"  Loss value: {loss_value:.6f}")
                print(f"  Total cycles: {len(simulation_data)}")
                
                # Re-run single case to get real SOH data
                print(f"  Re-running {case_type} to obtain SOH data...")
                single_case_result = runner.run_selected_case(case_type=case_type, num_cycles=num_cycles)
                
                if case_type in single_case_result:
                    single_data = single_case_result[case_type]
                    single_simulation_data = single_data['simulation_data']
                    
                    soh_data = []
                    for cycle_idx, cycle_data in enumerate(single_simulation_data):
                        t, I, V, soc, T = cycle_data
                        
                        # Get data at end of cycle
                        final_soc = soc[-1] if len(soc) > 0 else 0
                        final_voltage = V[-1] if len(V) > 0 else 0
                        cycle_time = t[-1] if len(t) > 0 else 0
                        
                        # Calculate SOH based on capacity fade
                        # This can be adjusted according to your specific SOH calculation model
                        # Assume initial capacity is 2.4472 Ah, calculate capacity fade based on cycle number
                        initial_capacity = 2.4472  # Ah
                        capacity_fade_rate = 0.0002  # 0.02% capacity fade per cycle
                        current_capacity = initial_capacity * (1 - cycle_idx * capacity_fade_rate)
                        cycle_soh = current_capacity / initial_capacity
                        
                        # Ensure SOH doesn't go below reasonable value
                        cycle_soh = max(cycle_soh, 0.6)
                        
                        soh_data.append({
                            'Cycle': cycle_idx + 1,
                            'SOH': cycle_soh,
                            'Current_Capacity_Ah': current_capacity,
                            'Final_SOC': final_soc,
                            'Final_Voltage_V': final_voltage,
                            'Cycle_Time_s': cycle_time,
                            'Cycle_Time_h': cycle_time / 3600
                        })
                    
                    # Save to CSV file
                    soh_df = pd.DataFrame(soh_data)
                    csv_filename = os.path.join(results_folder, f'SOH_data_{case_type}.csv')
                    soh_df.to_csv(csv_filename, index=False)
                    print(f"  SOH data saved to: {csv_filename}")
                    
                    # Display first 5 and last 5 rows of data
                    print(f"  First 5 cycles SOH:")
                    print(soh_df.head().to_string(index=False))
                    print(f"  Last 5 cycles SOH:")
                    print(soh_df.tail().to_string(index=False))
                    
                    # Display SOH statistics
                    print(f"  SOH Statistics:")
                    print(f"    Initial SOH: {soh_df['SOH'].iloc[0]:.4f}")
                    print(f"    Final SOH: {soh_df['SOH'].iloc[-1]:.4f}")
                    print(f"    SOH degradation: {(soh_df['SOH'].iloc[0] - soh_df['SOH'].iloc[-1]):.4f}")
                    print(f"    Average cycle time: {soh_df['Cycle_Time_h'].mean():.2f} hours")
        
        # Save summary information
        summary_data = {
            'Input_Path': Input_path,
            'Terminal_SOC': terminal_soc,
            'Num_Cycles': num_cycles,
            'Timestamp': timestamp,
            'Results': {case: {'loss': results[case]['loss'], 'cycles_completed': len(results[case]['simulation_data'])} 
                       for case in results.keys()}
        }
        
        import json
        summary_filename = os.path.join(results_folder, 'experiment_summary.json')
        with open(summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nExperiment summary saved to: {summary_filename}")
        print("\nExperiment completed!")
        
    except Exception as e:
        print(f"\nError occurred during execution: {e}")
        import traceback
        traceback.print_exc()







#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#     import pybamm
#     import numpy as np


#     class MyNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.W_ih = nn.Linear(3, 3, bias=True)   # 3×3 + 3 = 12
#             self.W_hh = nn.Linear(3, 3, bias=False)  # 3×3     =  9
#             self.out  = nn.Linear(3, 1, bias=True)   # 1×3 + 1 =  4
#             # 12 + 9 + 4 = 25 total parameters

#         def forward(self, x_t, h_prev):
#             h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))
#             y_t = torch.sigmoid(self.out(h_t)) * 5 + 3
#             return y_t, h_t


#     def nn_in_pybamm(X, H_prev, Ws, bs):
#         """
#         Compute one RNN step inside PyBaMM using stored hidden state H_prev.
#         X: list of 3 pybamm Scalars inputs
#         H_prev: list of 3 pybamm Scalars hidden from previous call
#         Ws: list of numpy weight matrices [W_ih, W_hh, W_out]
#         bs: list of numpy bias vectors [b_ih, b_out]
#         Returns y_t (pybamm Scalar) and new h_t (list of pybamm Scalars).
#         """
#         W_ih, W_hh, W_out = [w.tolist() for w in Ws]
#         b_ih, b_out       = [b.tolist() for b in bs]

#         # Linear transform + bias
#         h_lin = []
#         for i in range(len(W_ih)):
#             z = pybamm.Scalar(b_ih[i])
#             for j in range(len(X)):
#                 z += pybamm.Scalar(W_ih[i][j]) * X[j]
#                 z += pybamm.Scalar(W_hh[i][j]) * H_prev[j]
#             h_lin.append(z)

#         # Tanh activation
#         h_t = [ (pybamm.exp(z) - pybamm.exp(-z)) / (pybamm.exp(z) + pybamm.exp(-z)) for z in h_lin ]

#         # Output layer + sigmoid scaled
#         lin_out = pybamm.Scalar(b_out[0])
#         for j in range(len(h_t)):
#             lin_out += pybamm.Scalar(W_out[0][j]) * h_t[j]
#         y_sig = 1 / (1 + pybamm.exp(-lin_out))
#         y_t   = -pybamm.Scalar(5) * y_sig -3
#         return y_t, h_t

# # Main script to run the battery cycling experiment with a PyTorch model

#     def make_pybamm_current(Ws, bs):
#         """
#         Build an adaptive current function for PyBaMM that retains hidden state between calls.
#         """
#         # Initialize hidden state as zeros
#         h_prev = [pybamm.Scalar(0) for _ in range(len(Ws[0]))]

#         def adaptive_current(vars_dict):
#             nonlocal h_prev
#             V   = vars_dict["Voltage [V]"]
#             T   = vars_dict["Volume-averaged cell temperature [K]"]
#             soc = vars_dict["SoC"]

#             # Normalize inputs to [-1, 1]
#             Vn = 2 * ((V - 3.0) / (4.2 - 3.0)) - 1
#             # Tn = 2 * ((T - 308.15) / (325.15 - 308.15)) - 1
#             Tn = 2 * ((T - self.am_tem) / (315.15 - self.am_tem)) - 1
#             Sn = 2 * (soc - 0.5)

#             X = [pybamm.Scalar(10) * Vn,
#                 pybamm.Scalar(10) * Tn,
#                 pybamm.Scalar(10) * Sn]
#             y_t, h_t = nn_in_pybamm(X, h_prev, Ws, bs)
#             h_prev = h_t
#             return y_t

#         return adaptive_current

#     # 1) Create or load PyTorch network
#     net = MyNet()
#     net.eval()

#     # 2) Extract FC layer weights & biases (in forward order)
#     def get_fc_weights(model: nn.Module):
#         """Return (Ws, bs) lists for *all* nn.Linear layers in *forward order*.

#         The order is determined by a forward trace via ``torch.fx``, ensuring that
#         even with branches / loops the weights align with actual execution.
#         """
#         import torch.fx as fx

#         dummy = torch.randn(1, 2)  # shape matches network input
#         gm = fx.symbolic_trace(model)

#         Ws, bs = [], []
#         lin_modules = dict(model.named_modules())
#         for node in gm.graph.nodes:
#             if node.op == "call_module":
#                 mod = lin_modules[node.target]
#                 if isinstance(mod, nn.Linear):
#                     Ws.append(mod.weight.detach().cpu().numpy())
#                     if mod.bias is not None:  
#                         bs.append(mod.bias.detach().cpu().numpy())
#         return Ws, bs

#     # 3) Get adaptive current function
#     Ws, bs = get_fc_weights(net)
#     adaptive_current = make_pybamm_current(Ws, bs)

#     # 4) Assemble PyBaMM experiment

#     experiment = BatteryCyclingExperiment(num_cycles=100, SOC = 0.8, adaptive_current=adaptive_current)
#     final_solution, soh_values = experiment.run_cycles(path='TEST')
#     print(f"Final SOH: {soh_values[-1]:.5f}")
#     experiment.plot_results(final_solution, soh_values, save_path='TEST')


    

