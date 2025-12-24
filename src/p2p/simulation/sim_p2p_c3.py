import pybamm
import numpy as np
import matplotlib.pyplot as plt
import logging

import torch
import torch.nn as nn
import pickle

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import psutil


am_tem = 308.15

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

class BatteryCyclingExperiment:
    """Cycle‑life experiment that avoids reference cycles by forwarding **only** the
    last step‐solution between cycles.

    Key fix
    --------
    The previous implementation passed the *full* `pybamm.Solution` from the
    preceding cycle into `Simulation.solve(starting_solution=…)`.  Because each
    `Solution` already keeps references to every child solution, doing this for
    many cycles built a deeply nested reference graph that could not be freed
    by Python's garbage collector, leading to runaway RAM usage.  

    We now store **just the final sub‑solution** (`sol.sub_solutions[-1]`) of a
    cycle and pass that lightweight object forward.  This breaks the reference
    chain and keeps memory consumption roughly constant across thousands of
    cycles.
    """

    def __init__(self, num_cycles=30, SOC=0.9, adaptive_current=None):
        # Silence PyBaMM logging
        pybamm.set_logging_level("ERROR")

        # Experiment parameters
        self.steps = 0
        self.num_cycles = num_cycles
        self.charging_duration = 0.5  # h
        self.SOC = SOC
        self.adaptive_current = adaptive_current
        self.am_tem = 308.5  # K
        self.nominal_capacity_battery =  2.4472 # Ah 

        # --- Build model --------------------------------------------------
        self._build_model()

        # Solver setup (IDAKLU is recommended for degradation models)
        self.solver = pybamm.IDAKLUSolver()
        self.soh_solver = pybamm.IDAKLUSolver()

        # # Capacity bookkeeping
        # self.current_capacity = self.nominal_capacity_battery
        # self.model.variables["Delta capacity [A.h]"] = (
        #     self.nominal_capacity_battery - self.current_capacity
        # )
        # self.model.variables["SoC"] = 1 - (
        #     self.model.variables["Discharge capacity [A.h]"]
        #     + self.model.variables["Delta capacity [A.h]"]
        # ) / self.nominal_capacity_battery


    # ---------------------------------------------------------------------
    #                               Model
    # ---------------------------------------------------------------------
    def _build_model(self):
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
            }
        )

        # Parameters
        self.parameter_values = pybamm.ParameterValues("Ai2020")
        k = self.parameter_values["SEI reaction exchange current density [A.m-2]"]
        self.parameter_values[
            "SEI reaction exchange current density [A.m-2]"
        ] = k * 5
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

        # Custom capacity‑loss variable
        V = self.model.variables["Terminal voltage [V]"]
        Q_loss = pybamm.Variable("Q_loss")
        V_threshold = pybamm.Parameter(V_threshold_param)
        alpha = pybamm.Parameter(alpha_param)
        n = pybamm.Parameter(n_param)
        over_voltage = pybamm.maximum(V - V_threshold, 0)
        dQdt = alpha * (over_voltage ** n)
        self.model.rhs[Q_loss] = dQdt
        self.model.initial_conditions[Q_loss] = pybamm.Scalar(0)

        self.current_capacity = self.nominal_capacity_battery
        # Book‑keeping variables
        self.model.variables["Delta capacity [A.h]"] = (
            self.nominal_capacity_battery - self.current_capacity
        )
        self.model.variables["SoC"] = 1 - (
            self.model.variables["Discharge capacity [A.h]"]
            + self.model.variables["Delta capacity [A.h]"]
        ) / self.nominal_capacity_battery

        # 90 % SoC termination event used during charging
        self.soc_termination = pybamm.step.CustomTermination(
            name="SoC = 90%", event_function=self.soc_90_cutoff
        )

    # ------------------------------------------------------------------
    #                        Helper functions
    # ------------------------------------------------------------------
    def soc_90_cutoff(self, variables):
        """Termination when the instantaneous SoC reaches the target."""
        return self.SOC - variables["SoC"]

    def measure_soh_and_new_capacity(self, solution):
        """Compute SOH and log degradation."""
        Q_loss = solution["Q_loss"](solution.t[-1])
        soh = (self.current_capacity - Q_loss) / self.nominal_capacity_battery
        # Update model bookkeeping variables
        self.model.variables["Delta capacity [A.h]"] = (
            self.nominal_capacity_battery - self.current_capacity
        )
        self.model.variables["SoC"] = 1 - (
            self.model.variables["Discharge capacity [A.h]"]
            + self.model.variables["Delta capacity [A.h]"]
        ) / self.nominal_capacity_battery
        return soh

    # ------------------------------------------------------------------
    #                          Cycle parts
    # ------------------------------------------------------------------
    def run_part1(self, prev_step_solution=None):
        """Discharge (C/3) → rest → adaptive charge."""

        steps_part1 = (
            pybamm.step.current(value=5 / 3, termination="3.0V"),
            pybamm.step.voltage(value=3.0, termination="50mA"),
            pybamm.step.CustomStepExplicit(
                self.adaptive_current,
                duration=self.charging_duration * 3600,
                termination=["4.18V", self.soc_termination],
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

        if prev_step_solution is None:
            sol_1 = sim_1.solve(initial_soc=0.999)
        else:
            # 🔑 Pass only the *last* sub‑solution from the previous cycle
            sol_1 = sim_1.solve(starting_solution=prev_step_solution)

        # Diagnostics
        steps = sol_1.cycles[-1].steps
        for i, step in enumerate(steps):
            print(f"Step {i + 1} termination reason: {step.termination}")
        self.steps += 1
        last_step_duration = steps[-1].t[-1] - steps[-2].t[-1]
        return sol_1, float(last_step_duration)

    def run_part2(self, sol_1, last_step_duration):
        """Top‑up charge if Part 1 stopped before target SoC."""
        soc_part1 = sol_1["SoC"].data[-1]
        discharge_cap_after_discharge = (
            sol_1.cycles[-1].steps[0]["Discharge capacity [A.h]"].data[-1]
        )
        self.current_capacity = discharge_cap_after_discharge

        if soc_part1 < self.SOC:
            delta_soc = self.SOC - soc_part1
            delta_Q = self.nominal_capacity_battery * delta_soc
            leftover_hours = self.charging_duration - (last_step_duration / 3600)
            if leftover_hours * 3600 <= 0:
                print("No leftover time for Part 2, skipping.")
                self.steps += 1
                return sol_1

            i_charge_part2 = delta_Q / leftover_hours
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
            sol_2 = sim_2.solve(starting_solution=sol_1.sub_solutions[-1])

            self.steps += 1
            return sol_2
        else:
            self.steps += 1
            print("The custom current in Part 1 brought SoC ≥ target, skipping Part 2.")
            return sol_1

    def steps_part3(self, sol_2, rest_time):
        rest_step = (pybamm.step.current(value=0, duration=rest_time),)
        experiment_3 = pybamm.Experiment([rest_step])
        sim_3 = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=experiment_3,
            solver=self.solver,
        )
        sim_3._esoh_solver = self.soh_solver
        return sim_3.solve(starting_solution=sol_2.sub_solutions[-1])


    # ------------------------------------------------------------------
    #                           Cycle loop
    # ------------------------------------------------------------------
    def run_cycles(self, path=None):
        """Run all cycles, checkpointing every 10 cycles if requested."""
        cycle_lim = 100
        if self.num_cycles > cycle_lim and path is None:
            raise ValueError(
                "For cycles greater than 30, please provide a valid path for checkpoint storage."
            )

        final_solution = None  # for summary statistics
        prev_step_solution = None  # 🔑 only last step, avoids reference cycles
        soh_values = []

        for cycle_index in range(self.num_cycles):
            print(f"\n--- Cycle {cycle_index + 1} ---")
            memory_monitor.log_memory_status(f"Cycle {cycle_index+1} start")


            # Part 1
            sol_1, last_step_duration = self.run_part1(prev_step_solution)

            # Part 2
            sol_2 = self.run_part2(sol_1, last_step_duration)

            # Part 3 (rest)
            part2_charing_duration = sol_2.t[-1] - sol_1.t[-1]
            rest_time = (
                3600 * self.charging_duration + 300 - part2_charing_duration - last_step_duration
            )
            sol_3 = self.steps_part3(sol_2, rest_time)

            # Book‑keeping
            final_solution = sol_3
            prev_step_solution = sol_3.sub_solutions[-1]  # 🔑 memory‑light hand‑off

            soh_val = self.measure_soh_and_new_capacity(sol_2)
            soh_values.append(soh_val)
            print(f"Cycle {cycle_index + 1} completed - SOH: {soh_val:.5f}")

            # Diagnostics & checks
            cycle_soc = sol_2["SoC"].data[-1]
            if abs(cycle_soc - self.SOC) > 0.01:
                raise ValueError(
                    f"Ending SoC {cycle_soc:.3f} deviates from target {self.SOC} by >1 %."
                )

            if self.num_cycles > cycle_lim and (cycle_index + 1) % 10 == 0:
                checkpoint_folder = os.path.join(path, "checkpoint")
                os.makedirs(checkpoint_folder, exist_ok=True)
                fout = os.path.join(checkpoint_folder, "soh_values.pkl")
                with open(fout, "wb") as f:
                    pickle.dump(soh_values, f)
                print(f"Checkpoint saved at cycle {cycle_index + 1} to {fout}")
            
            memory_monitor.log_memory_status(f"Cycle {cycle_index+1} end")


        # Summary
        print("\n--- All cycles complete ---")
        print(f"Final SOC: {final_solution['SoC'].data[-1]:.4f}")
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

# Example usage:
if __name__ == "__main__":

    # def adaptive_current(vars_dict):
    #     import pybamm
    #     V   = vars_dict["Voltage [V]"]
    #     SOC = vars_dict["SoC"]
    #     T   = vars_dict["Volume-averaged cell temperature [K]"]

    #     dv = 4.2 - V
    #     soc_decay = pybamm.exp(-6.0 * SOC)                   # rapidly decays as SOC increases
    #     voltage_drive = pybamm.arctan(6.0 * dv) / pybamm.arctan(6.0 * 1.7)  # normalized over V range
    #     blend = (voltage_drive + soc_decay) / 2.0            # blend the two influences

    #     current = 5.0 * blend +3
    #     return current * (-1)
    
    def adaptive_current(vars_dict):
        V   = vars_dict["Voltage [V]"]
        T   = vars_dict["Volume-averaged cell temperature [K]"]
        SOC = vars_dict["SoC"]
        
        v_mod = (pybamm.cos((V - 3.6) * 3) + 1) / 2
        soc_mod = (pybamm.exp(-10 * (SOC - 0.8) * (SOC - 0.3))) 
        t_mod = (pybamm.tanh((T - 298.15) / 10) + 1) / 2

        blend = v_mod * soc_mod * t_mod
        current = 3 + 5 * blend
        return current*(-1)-3

    experiment = BatteryCyclingExperiment(num_cycles=100, SOC = 0.9, adaptive_current=adaptive_current)
    final_solution, soh_values= experiment.run_cycles(path='TEST')
    print(f"Final SOH: {soh_values[-1]:.5f}")
    experiment.plot_results(final_solution, soh_values, save_path='TEST')
    

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import ListedColormap

    # plot soc current and voltage with time 
    def plot_soc_current_voltage(solution, save_path=None):
        if save_path and not save_path.endswith(os.sep):
            save_path += os.sep

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot SoC
        ax1.plot(solution.t, solution["SoC"].data, label="SoC", color='blue')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("SoC", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for current
        ax2 = ax1.twinx()
        ax2.plot(solution.t, solution["Current [A]"].data, label="Current", color='red')
        ax2.set_ylabel("Current (A)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Create a third y-axis for voltage
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
        ax3.plot(solution.t, solution["Voltage [V]"].data, label="Voltage", color='green')
        ax3.set_ylabel("Voltage (V)", color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        fig.tight_layout()  # Adjust layout to prevent overlap
        plt.title("SoC, Current and Voltage over Time")
        
        plt.savefig(os.path.join(save_path, "soc_current_voltage_plot.png"))
        plt.close()

    # Example usage of the plot function
    plot_soc_current_voltage(final_solution, save_path='TEST')



