import os
import sys
import gc
import csv
import logging
import importlib.util
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pybamm
import psutil
import torch
import torch.nn as nn

# Add paths for local modules
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

am_tem = 308.15  # Ambient temperature in Kelvin (25°C)

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

# 创建全局内存监测器实例
memory_monitor = MemoryMonitor()

class BatteryCyclingExperiment_CCCV:
    def __init__(
        self,
        num_cycles: int,
        SOC: float,
        cc_rates: list[float],
        charging_duration: float = 0.5,  # hours
        initial_capacity: float = 2.4472,
    ):
        """
        num_cycles: Number of cycles to run
        SOC: Target state of charge (0-1)
        cc_rates: List of charge currents (A); use negative values for charging
        charging_duration: Maximum duration for each CC segment (hours)
        initial_capacity: Initial cell capacity (A·h)
        """
        pybamm.set_logging_level("ERROR")
        self.num_cycles = num_cycles
        self.SOC = SOC
        self.cc_rates = cc_rates
        self.charging_duration = charging_duration
        self.current_capacity = initial_capacity
        self.nominal_capacity_battery = 2.4472
        self.am_tem = 308.15  # Ambient temperature in Kelvin (25°C)

        # Base DFN model with SEI and porosity-change
        self.model = pybamm.lithium_ion.DFN(
            options={
                "particle": "Fickian diffusion",
                "particle mechanics": "swelling and cracking",  # other options are "none", "swelling only"
                # "loss of active material": "current-driven",
                "loss of active material": "stress-driven",

                "SEI": "reaction limited", 
                "SEI porosity change": "true",
                "calculate discharge energy": "true",
                "SEI on cracks": "true",

                "thermal": "lumped",  # Lumped thermal model (simplified)
                     },
                
            )

        self.parameter_values = pybamm.ParameterValues("Ai2020")
        k = self.parameter_values["SEI reaction exchange current density [A.m-2]"]
        self.parameter_values["SEI reaction exchange current density [A.m-2]"] = k*5
        self.parameter_values.update({
            # "Positive electrode current-driven LAM rate": current_LAM,
            # "Negative electrode current-driven LAM rate": current_LAM,
            "Negative electrode LAM constant proportional term [s-1]": 1e-12,
            "Positive electrode LAM constant proportional term [s-1]": 2.78e-13*10,
            "Positive electrode cracking rate":3.9e-20*10,
            "Negative electrode cracking rate":3.9e-20*10,
            "Total heat transfer coefficient [W.m-2.K-1]": 5.0, 
            "Ambient temperature [K]": self.am_tem,  #  (35 °C) 
            }, check_already_exists=False)

        # Overvoltage degradation
        self.parameter_values.update(
            {
                "Overvoltage threshold [V]": 4.2,
                "Overvoltage degradation factor [1/s]": 0.3,
                "Overvoltage exponent": 3,
            },
            check_already_exists=False,
        )
        self.parameter_values.set_initial_stoichiometries(1)

        # Add Q_loss variable and ODE
        V = self.model.variables["Terminal voltage [V]"]
        Q_loss = pybamm.Variable("Q_loss")
        V_th = pybamm.Parameter("Overvoltage threshold [V]")
        alpha = pybamm.Parameter("Overvoltage degradation factor [1/s]")
        n = pybamm.Parameter("Overvoltage exponent")
        over_volt = pybamm.maximum(V - V_th, 0)
        self.model.rhs[Q_loss] = alpha * over_volt ** n
        self.model.initial_conditions[Q_loss] = pybamm.Scalar(0)

        # Initialize SoC variables
        self._update_soc_vars()

        # Create termination events for each CC segment
        n_cc = len(cc_rates)
        fractions = [(i / (n_cc + 1)) * SOC for i in range(1, n_cc + 1)]
        self.cc_events = []
        for frac in fractions:
            event = pybamm.step.CustomTermination(
                name=f"SoC = {frac*100:.1f}%",
                event_function=lambda vars, thr=frac: thr - vars["SoC"],
            )
            self.cc_events.append(event)

        # Final SOC event for Part 2
        self.soc_target_event = pybamm.step.CustomTermination(
            name=f"SoC = {SOC*100:.1f}%",
            event_function=lambda vars: SOC - vars["SoC"],
        )
        keep = [
            "Voltage [V]",              
            "SoC",
            "Discharge capacity [A.h]",
            "Q_loss",
            # "Volume-averaged cell temperature [K]", 
        ]
        
        self.solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6, output_variables=keep)

    def _update_soc_vars(self):
        """Update Delta capacity and SoC definitions before each simulation."""
        self.model.variables["Delta capacity [A.h]"] = pybamm.Scalar(
            self.nominal_capacity_battery - self.current_capacity
        )
        self.model.variables["SoC"] = 1 - (
            self.model.variables["Discharge capacity [A.h]"]
            + self.model.variables["Delta capacity [A.h]"]
        ) / self.nominal_capacity_battery

    def run_part1(self, prev_solution=None):
        """Part 1: Discharge, voltage hold, then multiple CC segments."""
        steps = [
            pybamm.step.current(value=5/3, termination="3.0V"),
            pybamm.step.voltage(value=3, termination="50mA"),
        ]
        for rate, event in zip(self.cc_rates, self.cc_events):
            steps.append(
                pybamm.step.current(
                    value=rate,
                    duration=self.charging_duration * 3600,
                    termination=[event],
                )
            )
        exp = pybamm.Experiment([tuple(steps)])
        sim = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=exp,
            solver=self.solver,
        )

        # sim._esoh_solver = self.soh_solver
        sol = (
            sim.solve(initial_soc=0.999)
            if prev_solution is None
            else sim.solve(starting_solution=prev_solution)
        )
        executed = sol.cycles[-1].steps
        cc_steps = executed[-len(self.cc_rates):]
        durations = [float(s.t[-1] - s.t[0]) for s in cc_steps]
        print(f"CC step durations (s): {durations}")
        return sol, durations

    def run_part2(self, sol1, durations):
        """Part 2: Top-off charging to target SOC if needed."""
        soc1 = sol1["SoC"].data[-1]
        self.current_capacity = sol1.cycles[-1].steps[0]["Discharge capacity [A.h]"].data[-1]
        self._update_soc_vars()

        if soc1 < self.SOC:
            delta_soc = self.SOC - soc1
            delta_Q = self.nominal_capacity_battery * delta_soc
            t = sum(durations)
            rem_h = max(self.charging_duration - t / 3600, 0)
            if rem_h > 0:
                print(f"Part2: remaining time for charging: {rem_h:.2f} h")
                i_charge = delta_Q / rem_h if rem_h > 0 else delta_Q / self.charging_duration
            else:
                raise RuntimeError(f"run_part1 exceeded 2 hours: {t/3600:.2f} h")

            step = pybamm.step.current(
                value=-i_charge,
                duration=self.charging_duration * 3600,
                termination=[self.soc_target_event],
            )
            exp = pybamm.Experiment([tuple([step])])
            sim = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_values,
                experiment=exp,
                solver=self.solver,
            )
            # sim._esoh_solver = self.soh_solver
            sol2 = sim.solve(starting_solution=sol1)
            return sol2

        print("Target SOC reached in Part 1; skipping Part 2.")
        return sol1

    def run_rest(self, sol2, rest_time):
        """Part 3: Rest at 0 A for 1 h."""
        step = pybamm.step.current(value=0, duration=rest_time)
        exp = pybamm.Experiment([tuple([step])])
        sim = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=exp,
            solver=self.solver,
        )
        # sim._esoh_solver = self.soh_solver
        sol3 = sim.solve(starting_solution=sol2)
        return sol3

    def measure_soh(self, sol):
        """Compute SOH from accumulated Q_loss (up to end of Part2)."""
        Q_loss_val = sol["Q_loss"](sol.t[-1])
        soh = (self.current_capacity - Q_loss_val) / self.nominal_capacity_battery
        # soh = self.current_capacity / self.nominal_capacity_battery
        print(f"Cycle SOH: {soh * 100:.2f}%")
        return soh

    def run_cycles(self):
        prev_solution = None
        soh_list = []
        
        for idx in range(self.num_cycles):
            print(f"\n--- Cycle {idx+1} ---")
            # 记录每个循环开始时的内存使用情况
            memory_monitor.log_memory_status(f"Cycle {idx+1} start")
            
            sol1, durations = self.run_part1(prev_solution)          
            sol2 = self.run_part2(sol1, durations)

            t0 = sol1.t[-1]
            t1 = sol2.t[-1]
            part2_charing_duration = t1 - t0
            rest_time = 300
            sol3 = self.run_rest(sol2, rest_time=rest_time)
            soh = self.measure_soh(sol2)
            soh_list.append(soh)
            
            # 检查 SOH 是否低于 70%
            # 原逻辑：SOH < 0.7 时直接停止。现改为仅警告，继续运行以记录所有结果。
            if soh < 0.7:
                print(
                    f"警告：SOH 降至 {soh:.3f} ({soh*100:.1f}%)，但继续运行以记录所有结果。"
                )
        
            # 只保留最后一个 sub-solution 作为下一轮的起点，避免引用环
            prev_solution = sol3.sub_solutions[-1] if sol3.sub_solutions else sol3
            
            # 记录每个循环结束时的内存使用情况
            memory_monitor.log_memory_status(f"Cycle {idx+1} end")
            
            # 主动清理当前循环的中间变量
            del sol1, sol2
            memory_monitor.cleanup_memory()
            
        return sol3, soh_list
    



    def plot_results(self, final_solution, soh_values, save_path="results"):
        os.makedirs(save_path, exist_ok=True)

        # Discharge capacity trend
        plt.figure()
        plt.plot(
            final_solution["Discharge capacity [A.h]"].data,
            label="Discharge capacity",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Capacity (A·h)")
        plt.title("Discharge Capacity Trend")
        plt.legend()
        plt.savefig(os.path.join(save_path, "discharge_capacity.png"))
        plt.close()

        # Current & Voltage
        plot = pybamm.QuickPlot(final_solution, ["Current [A]", "Voltage [V]"])
        plot.plot(0)
        plt.savefig(os.path.join(save_path, "current_voltage.png"))
        plt.close()

        # SOH progression
        plt.figure()
        plt.plot(range(1, self.num_cycles + 1), soh_values, marker="o")
        plt.xlabel("Cycle")
        plt.ylabel("SOH")
        plt.title("SOH Over Cycles")
        plt.savefig(os.path.join(save_path, "soh_progression.png"))
        plt.close()


def plot_protocol_with_loss(save_path, combined_simulation_data, combined_loss_data, terminal_soc=0.9, type='nn',custom_colors=None):
    """
    绘制电流、电压、SoC、温度四张子图，并根据 loss 值从深蓝到浅蓝为每个 case 分配颜色。
    使用 inset_axes 创建一个小型 colorbar，显示排序后的 SoH 值（反推自 loss）。
    """
    # 1. 根据 loss 值排序：loss 较低的在前（表现好）
    sorted_cases = sorted(combined_loss_data, key=combined_loss_data.get)  # 例如：['case1_Best', 'case2_Best', 'case1_Middle', ...]
    n = len(sorted_cases)

    base_colors = ["#0747a1", "#cce6ff"]  
    cmap = LinearSegmentedColormap.from_list("custom_blue", base_colors, N=n)

    # 2. 如果没有传入 custom_colors，则按默认逻辑生成
    if custom_colors is None:

        # 分配颜色
        colors = {}
        for i, case in enumerate(sorted_cases):
            if i == 0:
                colors[case] = "#043178"  # 第一名颜色单独加深
            else:
                colors[case] = mcolors.to_hex(cmap(i / (n - 1)))

    else:
        colors = custom_colors

    
    color_list = [colors[case] for case in sorted_cases]
    
    # 4. 建立 4 个子图（电流、电压、SoC、温度）
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6, 10))
    ax_current, ax_voltage, ax_soc, ax_temp = axes

    # 5. 添加参考线
    ax_current.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax_voltage.axhline(4.2, linestyle="--", linewidth=1, color="gray")
    ax_voltage.axhline(3.0, linestyle="--", linewidth=1, color="gray")
    ax_soc.axhline(terminal_soc, linestyle="--", linewidth=1, color="gray")
    ax_soc.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    ax_temp.axhline(am_tem, linestyle="--", linewidth=1, color="gray")  # 常温参考线 (25°C)
    
    # 6. 假设所有 case 共享相同的时间间隔（从任意一个 case 中取样）
    sample_case = list(combined_simulation_data.values())[-1]

    # 根据数据长度确定是否包含温度数据
    if len(sample_case) == 4:
        t_common, I_common = sample_case[0], sample_case[1]
    else:
        t_common, I_common = sample_case[0], sample_case[1]
    
    t_common_in_hours = t_common / 3600

    # 7. 确定充电/放电区域（示例：当电流小于 -0.01 为充电区域）
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

    # 8. 对所有子图添加区域阴影
    for interval in charging_intervals_in_hours:
        for ax in axes:
            ax.axvspan(*interval, color="#eddca5", alpha=0.3)
    for interval in discharging_intervals_in_hours:
        for ax in axes:
            ax.axvspan(*interval, color="lightgray", alpha=0.3)

    # 9. 绘制每个 case 的曲线，使用分配的颜色
    for case, data in combined_simulation_data.items():
        # 兼容原有4元素数据 (t, I, V, soc) 和新的5元素数据 (t, I, V, soc, T)
        if len(data) == 4:
            t, I, V, soc = data
            T = None  # 如果没有温度数据，则不绘制温度曲线
        else:
            t, I, V, soc, T = data  # data 应为 (t, I, V, soc, T)
            
        t_in_hours = t / 3600
        color = colors.get(case, "black")
        ax_current.plot(t_in_hours, I, linestyle="-", color=color, 
                        label=f"{case} (loss: {combined_loss_data[case]:.2f})", alpha=0.8)
        ax_voltage.plot(t_in_hours, V, linestyle="-", color=color, alpha=0.9)
        ax_soc.plot(t_in_hours, soc, linestyle="-", color=color, alpha=0.8)
        
        # 如果有温度数据，则绘制温度曲线
        if T is not None:
            ax_temp.plot(t_in_hours, T, linestyle="-", color=color, alpha=0.8)

    # 10. 添加各子图的辅助文字
    ax_current.text(0.72, 0.8, 'Charging', transform=ax_current.transAxes, fontsize=11, color='gray', alpha=0.6)
    ax_current.text(0.22, 0.8, 'Discharging', transform=ax_current.transAxes, fontsize=11, color='gray', alpha=0.6)
    ax_voltage.text(t_common_in_hours[-1]*0.8, 4.2, '4.2V', va='top', ha='right', fontsize=10, color='gray')
    ax_voltage.text(t_common_in_hours[-1]*0.95, 3.0, '3.0V', va='bottom', ha='right', fontsize=10, color='gray')
    ax_soc.text(t_common_in_hours[-1]*0.95, terminal_soc, f'{terminal_soc*100:.0f}% SoC', va='top', ha='right', fontsize=10, color='gray')
    ax_soc.text(t_common_in_hours[-1]*0.95, 0, '0% SoC', va='bottom', ha='right', fontsize=10, color='gray')
    ax_temp.text(t_common_in_hours[-1]*0.95, am_tem, '25°C', va='top', ha='right', fontsize=10, color='gray')

    # 11. 添加示意箭头（根据需要调整）
    if charging_intervals_in_hours:
        ax_current.annotate(
            '', 
            xy=(charging_intervals_in_hours[0][0], 1.3), 
            xytext=(charging_intervals_in_hours[0][1], 1.3),
            arrowprops=dict(arrowstyle='<->', color='black')
        )
        ax_current.text(
            (charging_intervals_in_hours[0][0] + charging_intervals_in_hours[0][1]) / 2, 1.35, 
            f'0.5h 0-{terminal_soc*100:.0f}% SoC', 
            ha='center', va='bottom', fontsize=8, color='black'
        )

    # 12. 设置标题与坐标轴标签
    if type == 'nn':
        fig.suptitle("Battery Protocols Optimized by Neural Network Based on SOH", fontsize=11, fontweight='bold', ha='center')
    else:
        fig.suptitle("Multi-Step CCCV Battery Protocols Based on SOH", fontsize=11, fontweight='bold', ha='center')
    ax_current.set_ylabel("Current [A]")
    ax_voltage.set_ylabel("Voltage [V]")
    ax_soc.set_ylabel("SoC")
    ax_temp.set_ylabel("Temperature [K]")
    ax_temp.set_xlabel("Time [h]")
    
    # 13. 创建一个小型的 inset colorbar，用于表示从深色到浅色对应的 SoH（反推自 loss）
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 在 ax_current 内创建一个小的 colorbar 区域，放在左下角
    cax = inset_axes(ax_current, width="38%", height="3%", loc='lower left', borderpad=3)
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    
    # 计算 colorbar 上各刻度对应的实际 SoH 值
    # 对应公式: SoH = 0.93 + 0.07 * exp(-loss)
    cmap_discrete = ListedColormap(color_list)

    # 4) 再创建 ScalarMappable，让 colorbar 使用这个离散色卡
    norm = mpl.colors.Normalize(vmin=0, vmax=n - 1)  # 或者 0~1，看你想怎么刻度
    sm = mpl.cm.ScalarMappable(cmap=cmap_discrete, norm=norm)
    sm.set_array([])

    # 5) 绘制 colorbar
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks(range(n))  # 离散刻度
    # 假设你想把 SoH 值对应到每一个 case 上
    soh_values_sorted = [0.93 + 0.07 * np.exp(-combined_loss_data[case]) for case in sorted_cases]
    soh_loss = (1 - np.array(soh_values_sorted))*100

    def format_half_up(value, ndigits=2):
        """
        对 value 做传统意义上的四舍五入（ROUND_HALF_UP），保留 ndigits 位小数，
        返回字符串，不带多余的零以外位数（始终保留 ndigits 位小数）。
        """
        # 构造 0.01、0.001 这样的量纲
        quantize_str = '0.' + '0'*(ndigits-1) + '1'
        d = Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
        return f"{d:.{ndigits}f}"
    
    labels = [ format_half_up(soh_val, 2) for soh_val in soh_loss ]

    cbar.set_ticklabels(labels)
    cbar.set_label("SoH Reduction %", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    # 14. 调整布局、保存并显示图像
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)




class BatteryExperimentRunner:
    def __init__(self, input_folder, terminal_soc=0.8, num_cycles=2, num_cc=35, type_name='nn'):
        """
        Initializes the runner with an input folder that contains:
        - A model_architecture.py file defining 'NeuralNetwork'.
        - A single 'all_params.pth' with stored parameters.
        """
        self.input_folder = input_folder
        # self.model = self.load_model()
        self.csv_rows = self.load_csv()
        self.terminal_soc = terminal_soc
        self.type_name = type_name
        self.num_cc = num_cc
        self.num_cycles = num_cycles

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

    def apply_parameters_cccv(self, param_index):
        """
        Loads parameters from 'all_params.pth' by index and returns a list
        [cc1, cc2, …, ccN] of length self.num_cc.
        If stored as dict, expects keys 'cc1'..'cc{num_cc}'.  
        If stored as list/tuple, expects at least self.num_cc entries.
        Empty strings are replaced with 0.0.
        """
        import os, torch

        pth_file = os.path.join(self.input_folder, 'all_params.pth')
        if not os.path.exists(pth_file):
            raise FileNotFoundError(f"{pth_file!r} not found.")

        stored = torch.load(pth_file)
        if param_index < 0 or param_index >= len(stored):
            raise IndexError(f"Index {param_index} out of range.")

        raw = stored[param_index]
        def parse(x):
            if isinstance(x, str) and x.strip()=="":
                return 0.0
            return float(x)

        # dict case
        if isinstance(raw, dict):
            # verify keys
            missing = [i for i in range(1, self.num_cc+1)
                       if f"cc{i}" not in raw]
            if missing:
                raise KeyError(f"Missing keys: {['cc'+str(i) for i in missing]}")
            cc_rates = [parse(raw[f"cc{i}"]) for i in range(1, self.num_cc+1)]

        # list/tuple case
        elif isinstance(raw, (list, tuple)) and len(raw) >= self.num_cc:
            cc_rates = [parse(raw[i-1]) for i in range(1, self.num_cc+1)]

        else:
            raise ValueError(
                "Stored params must be dict with cc1..ccN or list/tuple of length ≥ N."
            )

        print(f"Loaded cc_rates (len={len(cc_rates)}): {cc_rates}")
        return cc_rates

    def run_experiment_cccv(self, param_index, case_name,
                            t_start=0, t_end=7200, num_points=1000):
        """
        Runs a CCCV experiment using the parameter set at 'param_index'
        (now a list of length self.num_cc) and saves plots/results.
        """
        # get a list [cc1, …, ccN]
        cc_rates = self.apply_parameters_cccv(param_index)

        # pass the entire list into your experiment
        experiment = BatteryCyclingExperiment_CCCV(
            num_cycles=self.num_cycles,
            SOC=self.terminal_soc,
            cc_rates=cc_rates,
            charging_duration=0.5,
        )
        final_solution, soh_values = experiment.run_cycles()

        # 保存输出结果
        output_folder = os.path.join(self.input_folder, "output", case_name)
        experiment.plot_results(final_solution, soh_values, save_path=output_folder)
        return final_solution, soh_values

    def run_experiment(self, param_index, case_name, t_start=0, t_end=7200, num_points=1000):
        """
        Runs an experiment by:
          - Applying parameters (by index) from 'all_params.pth' to the model.
          - Generating a drive cycle current.
          - Running the battery cycling simulation.
          - Saving the results (plots, etc.) to an output folder.
        """
        self.model = self.load_model()
        # Apply the selected parameters
        self.apply_parameters(param_index)

        # Generate time values and obtain model output
        t_values = np.linspace(t_start, t_end, num_points).reshape(-1, 1)
        t_tensor = torch.tensor(t_values, dtype=torch.float32)
        output = self.model(t_tensor).detach().numpy()
        output = -output  # Invert the output if needed
        data = np.column_stack((t_values, output))

        # Run the battery cycling experiment
        experiment = BatteryCyclingExperiment(num_cycles=self.num_cycles, SOC=self.terminal_soc, drive_cycle_current=data)
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

        # 过滤掉 loss 为 1e6 的记录
        valid_rows_with_index = [
            (i, row) for i, row in enumerate(self.csv_rows)
            if float(row["total_loss"]) != 1e6
        ]
        if not valid_rows_with_index:
            raise ValueError("No valid rows with loss ≠ 1e6 found.")

        # 排序并取 best, worst, middle
        rows_sorted = sorted(valid_rows_with_index, key=lambda x: float(x[1]["total_loss"]))
        keys = ["Best", "Worst", "Middle"]
        indices = [
            rows_sorted[0][0],
            rows_sorted[len(rows_sorted) // 4][0],
            rows_sorted[len(rows_sorted) // 2][0]
        ]
        loss_values = {
            keys[i]: float(rows_sorted[j][1]["total_loss"])
            for i, j in enumerate([0, -1, len(rows_sorted) // 2])
        }

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
                # 提取全部周期的数据
                t = sol["Time [s]"].data
                I = sol["Current [A]"].data
                V = sol["Voltage [V]"].data
                soc = sol["SoC"].data
                
                # 尝试提取温度数据
                try:
                    # PyBaMM中温度的常见变量名
                    temp_keys = ["Volume-averaged cell temperature [K]"]
                    T = None
                    for temp_key in temp_keys:
                        try:
                            T = sol[temp_key].data
                            break
                        except (KeyError, AttributeError):
                            continue
                    
                except Exception as e:
                    print(f"警告：温度提取失败")
                simulation_data[case_name] = (t, I, V, soc, T)
                all_sohs[case_name] = soh  # soh aligned with t
            except Exception as e:
                print(f"Error in {case_name}: {e}")

        # 运行三种 case
        for key, idx in zip(keys, indices):
            safe_run_experiment(idx, key)

        # —— 以下为新增：切出第二圈数据 —— #
        for case_name, sim_data in simulation_data.items():
            # 兼容4个或5个元素的数据结构
            if len(sim_data) == 4:
                t, I, V, soc = sim_data
                T = None
            else:
                t, I, V, soc, T = sim_data
            
            # 找 boundary：第一个从 ≤0 跳到 >0 的索引
            flip_idxs = np.where((I[:-1] <= 0) & (I[1:] > 0))[0]
            if flip_idxs.size == 0:
                raise RuntimeError(f"{case_name}: 未找到 ≤0→>0 跳变，无法切分第二圈")
            boundary = flip_idxs[0] + 1

            # 安全检查
            if boundary >= len(t):
                raise RuntimeError(f"{case_name}: 跳变索引 {boundary} 超出数组长度 {len(t)}")

            # 切片：从 boundary 到末尾即第二圈
            t2   = t[boundary:]   - t[boundary]    # 可选：时间重置为从0开始
            I2   = I[boundary:]
            V2   = V[boundary:]
            soc2 = soc[boundary:]
            soh2 = all_sohs[case_name][boundary:]

            # 覆盖原数据 - 保持原有的数据结构
            if T is not None:
                T2 = T[boundary:]
                simulation_data[case_name] = (t2, I2, V2, soc2, T2)
            else:
                simulation_data[case_name] = (t2, I2, V2, soc2)
            all_sohs[case_name] = soh2

        # 绘制汇总 SOH 曲线（如果需要）
        self.plot_aggregate_soh(all_sohs)

        # 合并返回
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
            'Worst': sorted_rows[-1][0],
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
        discharge_capacity_all = sol['Discharge capacity [A.h]'].data
        
        # 尝试提取温度数据
        try:
            # PyBaMM中温度的常见变量名
            temp_keys = ["Volume-averaged cell temperature [K]"]
            T_all = None
            for temp_key in temp_keys:
                try:
                    T_all = sol[temp_key].data
                    break
                except (KeyError, AttributeError):
                    continue
            
            if T_all is None:
                # 如果找不到温度数据，生成基于物理模型的模拟温度
                T_base = self.am_tem  # 25°C基础温度
                heat_from_current = np.abs(I_all) * 0.5  # 电流产生的热量
                T_all = T_base + heat_from_current + np.random.normal(0, 0.5, len(I_all))
                print(f"警告：{case_type} 未找到温度数据，使用模拟温度")
        except Exception as e:
            # 如果提取失败，生成模拟温度
            T_base = self.am_tem
            heat_from_current = np.abs(I_all) * 0.5
            T_all = T_base + heat_from_current + np.random.normal(0, 0.5, len(I_all))
            print(f"警告：{case_type} 温度提取失败，使用模拟温度: {e}")

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
            T_seg = T_all[start:end] if T_all is not None else None
            discharge_capacity_seg = discharge_capacity_all[start:end]
            if T_seg is not None:
                simulation_slices.append((t_seg, I_seg, V_seg, soc_seg, T_seg))
            else:
                simulation_slices.append((t_seg, I_seg, V_seg, soc_seg))


        return {
            case_type: {
                'simulation_data': simulation_slices,
                'loss': loss_val
            }
        }


    




# Example usage
if __name__ == "__main__":
    # def collect_results(input_paths, prefix, type_name, terminal_soc):
    #     combined_sim = {}
    #     combined_loss = {}
    #     for idx, path in enumerate(input_paths, start=1):
    #         runner = BatteryExperimentRunner(
    #             path,
    #             terminal_soc=terminal_soc,
    #             num_cycles=2,
    #             num_cc=3,
    #             type_name=type_name
    #         )
    #         data = runner.run_best()
    #         for key in ["Best", "Worst", "Middle"]:
    #             combined_sim[f"{prefix}_case{idx}_{key}"]  = data[key]["simulation_data"]
    #             combined_loss[f"{prefix}_case{idx}_{key}"] = data[key]["loss"]
    #     return combined_sim, combined_loss

    # Input_path =['CCCV/CCCVResults_3cc_newbattery']
    # terminal_soc = 0.9
    # type_name = 'cccv'  # or 'cccv'

    # # Input_path = ['Degradation/20250424_232208_model_0', 'Degradation/20250424_232208_model_1']
    # # terminal_soc = 0.8
    # # type_name = 'nn'  # or 'cccv'

    # combined_sim_data, combined_loss_data = collect_results(
    #     Input_path, "Single", type_name, terminal_soc
    # )

    # plot_protocol_with_loss("Save_figures/cccv_308tem_5_craking_90%.png", combined_sim_data, combined_loss_data, terminal_soc=terminal_soc)
    # print("Plot saved as protocol_plot.png")

    experiment = BatteryCyclingExperiment_CCCV(
        num_cycles=100,
        SOC=0.8,
        cc_rates=[-3.0, -4.2, -6.1],
        charging_duration=0.5,  # 修改为0.5小时
    )
    final_solution, soh_values = experiment.run_cycles()
    experiment.plot_results(final_solution, soh_values, save_path="TEST_CCCV_results")

    # Report any voltages > 4.2 V
    Vmax = final_solution["Terminal voltage [V]"].data
    exceeds = [v for v in Vmax if v > 4.2]
    print("Voltages > 4.2 V:", exceeds)
