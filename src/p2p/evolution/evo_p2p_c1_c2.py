# -*- coding: utf-8 -*-
"""battery_evolution.py
---------------------------------
Evolves battery-charging algorithms using an LLM-assisted genetic algorithm.
The loss calculation (SOH-based, –log mapping with < 0.93 hard penalty) is
left intact, per user request; other parts were refactored for robustness.

*Fixes in this revision*
-----------------------
- Ensure **all 10 initial algorithms** are extracted and executed.
- Skip empty / invalid code blocks; log explicitly.
- Child extraction picks the **first valid block** in response.
- Removed duplicate child evaluations.
- **Single, stable results root across processes** via CONTROL_RESULTS_DIR.
- **Per-generation subfolder** Gen_{k} directly under the results root.
"""

from __future__ import annotations

import datetime as _dt
import multiprocessing as mp
import random
import re
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ax.core.metric import Metric

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate_model.ECM_gradient_descent import ECMLayer

try:  # Optional dependency kept for legacy algorithms that still import pybamm
    import pybamm  # type: ignore
except ImportError:
    pybamm = None

from llm_generation_case1 import generate_initialization, generate_new_algorithm

# ───────────────────────── Global config ──────────────────────────
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)

# Target heat mode: "function" (default irregular wave) or "constant" (value = 5)
TARGET_HEAT_MODE = "constant"  # Change to "constant" to use fixed value of 5

# Use spawn for PyTorch + multiprocessing compatibility
_MP_CTX = mp.get_context("spawn")

# RegEx: fenced code blocks (``` or ```python)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# ───────────────── Results root (stable across processes) ─────────────────
# Parent fixes the root once; children reuse via env CONTROL_RESULTS_DIR
_env_dir = os.environ.get("CONTROL_RESULTS_DIR")
if _env_dir:
    _RESULTS_DIR = Path(_env_dir)
else:
    _TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _RESULTS_DIR = Path(f"./Control_simulation_results_case1cons_{_TIMESTAMP}").resolve()
    os.environ["CONTROL_RESULTS_DIR"] = str(_RESULTS_DIR)

_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_GEN_LOG = _RESULTS_DIR / "generation_log.csv"

# ───────────────────────── Helper utilities ──────────────────────────
def _extract_all_code(markdown: str) -> List[str]:
    """Return **all** fenced code blocks. Fallback to raw text if none."""
    blocks = _CODE_BLOCK_RE.findall(markdown)
    if not blocks:
        return [markdown]
    return blocks

def _first_valid_code(markdown: str) -> str:
    """Pick first block that defines `current_function`, else first block."""
    for blk in _extract_all_code(markdown):
        if "def current_function" in blk:
            return blk
    return _extract_all_code(markdown)[0]

def _save_algo(code: str, dest: Path) -> Callable:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(code)
    ns: Dict[str, object] = {
        "__name__": "algorithm_runtime",
        "__builtins__": __builtins__,
        "np": np, "numpy": np,
        "pd": pd, "pandas": pd,
        "torch": torch,
        "random": random,
    }
    if pybamm is not None:
        ns["pybamm"] = pybamm
    exec(code, ns, ns)
    func = ns.get("current_function")
    if not callable(func):
        raise ValueError("current_function() missing or not callable")
    return func

# ───────────────────────── Population initialisation ──────────────────────────
def initialize_population() -> List[Dict]:
    print("[Init] Generating initial algorithms via LLM…")
    response = generate_initialization()
    code_blocks = _extract_all_code(response)

    population: List[Dict] = []
    for idx, code in enumerate(code_blocks):
        indiv_dir = _RESULTS_DIR / f"Individual_{idx}"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        algo_path = indiv_dir / "algorithm.py"
        try:
            if not code.strip():
                raise ValueError("Empty code block")
            func = _save_algo(code, algo_path)
            loss = RunCharging(
                result_folder=indiv_dir,
                charging_current=func,
                target_heat_mode=TARGET_HEAT_MODE
            ).run_simulation({})
            print(f"[Init] Individual {idx} evaluated – loss {loss:.4f}")
        except Exception:
            traceback.print_exc()
            loss = 1e6
            print(f"[Init] Individual {idx} invalid – loss set to 1e6")
        population.append({
            "id": idx,
            "folder": indiv_dir,
            "file": algo_path,
            "loss": loss,
        })
    return population

# ───────────────────────── Genetic operators ──────────────────────────
def _tournament(pop: List[Dict], k: int = 3) -> Dict:
    return min(random.sample(pop, k), key=lambda d: d["loss"])

# ───────────────────────── Multiprocessing task ──────────────────────────
def _spawn_child(task):
    gen, child_idx, p1_file, p2_file = task

    # Each generation has its own subfolder directly under results root
    generation_dir = _RESULTS_DIR / f"Gen_{gen}"
    generation_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"gen{gen}_child{child_idx}"
    algo_path = generation_dir / f"{run_id}.py"
    result_path = generation_dir / f"results_{run_id}.csv"

    try:
        p1_code = Path(p1_file).read_text()
        p2_code = Path(p2_file).read_text()

        child_markdown = generate_new_algorithm(p1_code, p2_code)
        child_code = _first_valid_code(child_markdown)

        func = _save_algo(child_code, algo_path)
        loss = RunCharging(
            result_folder=generation_dir,
            charging_current=func,
            run_id=run_id,
            target_heat_mode=TARGET_HEAT_MODE
        ).run_simulation({})
    except Exception:
        traceback.print_exc()
        loss = 1e6
        # leave a placeholder for debugging
        try:
            algo_path.write_text("# error placeholder\n")
        except Exception:
            pass

    return {
        "Generation": gen,
        "Child_Number": child_idx,
        "Parent1": p1_file,
        "Parent2": p2_file,
        "Child_Folder": str(generation_dir),
        "Child_File": str(algo_path),
        "Child_Loss": loss,
        "Child_Result_File": str(result_path),
        "Timestamp": _dt.datetime.now().isoformat(),
    }

# ───────────────────────── CSV logging ──────────────────────────
def _append_logs(rows: List[Dict]):
    df_new = pd.DataFrame(rows)
    if _GEN_LOG.exists():
        df_new = pd.concat([pd.read_csv(_GEN_LOG), df_new], ignore_index=True)
    df_new.to_csv(_GEN_LOG, index=False)

# ───────────────────────── Simulation metric ──────────────────────────
class RunCharging(Metric):
    """Evaluate charging policies with the lightweight ECM used in nn_evolution_EA."""

    def __init__(
        self,
        name="total_loss",
        *,
        result_folder: Path,
        charging_current: Callable,
        run_id: Optional[str] = None,
        target_heat_mode: str = "function",  # "function" or "constant"
        **kw,
    ):
        super().__init__(name=name, **kw)
        self.result_folder = result_folder
        self.charging_current = charging_current
        self.run_id = run_id
        self.target_heat_mode = target_heat_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ecm_layer = ECMLayer().to(self.device)
        self.t_values = np.linspace(0, 3600, 36)
        self.T_initial = torch.tensor(25.0, device=self.device)
        self.target_heat = self._compute_target_heat()

    @staticmethod
    def loss_function(predicted_heat: torch.Tensor, target_heat: torch.Tensor) -> torch.Tensor:
        return torch.mean((predicted_heat - target_heat) ** 2)

    def _compute_target_heat(self) -> torch.Tensor:
        """
        Compute target heat profile.
        - If target_heat_mode == "constant": returns constant value of 5
        - If target_heat_mode == "function": uses the irregular wave function
        """
        if self.target_heat_mode == "constant":
            # Return constant value of 5 for all time steps
            return torch.full(
                (len(self.t_values),),
                5.0,
                device=self.device,
                dtype=torch.float32
            )
        else:  # "function" mode (default)
            # Original irregular wave function
            def irregular_wave(t):
                return 5 * np.sin(0.0005 * t) + 3 * np.sin(0.003 * t)

            irregular_current_values = torch.tensor(
                [irregular_wave(t) for t in self.t_values],
                device=self.device,
                dtype=torch.float32,
            )

            T = self.T_initial.clone()
            target_heat_irregular = []
            with torch.no_grad():
                for I in irregular_current_values:
                    Q, _, T = self.ecm_layer(I, T)
                    target_heat_irregular.append(Q)

            return torch.stack(target_heat_irregular)

    def _evaluate_currents(self) -> torch.Tensor:
        """Call the LLM-generated policy and coerce its output into a current profile."""
        func = self.charging_current
        try:
            raw_output = func(self.t_values)
        except TypeError:
            raw_output = func()

        if callable(raw_output):
            raw_output = [raw_output(float(t)) for t in self.t_values]
        elif hasattr(raw_output, "evaluate"):
            raw_output = [float(raw_output.evaluate(t=float(t))) for t in self.t_values]
        elif isinstance(raw_output, torch.Tensor):
            raw_output = raw_output.detach().cpu().numpy()
        elif np.isscalar(raw_output):
            raw_output = np.full_like(self.t_values, float(raw_output), dtype=float)

        if isinstance(raw_output, np.ndarray):
            raw_output = raw_output.tolist()

        if not isinstance(raw_output, (list, tuple)):
            raise ValueError("Charging policy must return a sequence of current values.")
        if len(raw_output) != len(self.t_values):
            raise ValueError(f"Expected {len(self.t_values)} current values, received {len(raw_output)}.")

        try:
            currents = torch.tensor(raw_output, dtype=torch.float32, device=self.device)
        except Exception as exc:  # defensive cast
            raise ValueError("Current sequence must be numeric.") from exc
        return currents

    def _record(self, loss, metrics):
        df = pd.DataFrame([{
            "loss": loss,
            "total_loss": metrics.get("total_loss", loss),
            "charging_time": metrics.get("charging_time", 0),
            "temperature": metrics.get("temperature", 0),
            "voltage": metrics.get("voltage", 0),
            "SOC": metrics.get("SOC", 0),
            "heat": str(metrics.get("heat", "None")),  # for CSV compatibility
            "target_heat": metrics.get("target_heat", 0.0)
        }])
        csv_name = f"results_{self.run_id}.csv" if self.run_id else "results.csv"
        csv_path = self.result_folder / csv_name
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    def run_simulation(self, _):
        try:
            current_profile = self._evaluate_currents()
        except Exception as exc:
            print(f"Error preparing current profile: {exc}")
            traceback.print_exc()
            return 1e6

        with torch.no_grad():
            T = self.T_initial.clone()
            predicted_heat = []
            for I in current_profile:
                Q, _, T = self.ecm_layer(I, T)
                predicted_heat.append(Q)
            predicted_heat_tensor = torch.stack(predicted_heat)
            loss = self.loss_function(predicted_heat_tensor, self.target_heat)

        total_loss = loss.item()
        metrics = {
            "total_loss": total_loss,
            "charging_time": float(self.t_values[-1]),
            "time_source": "fixed",
            "temperature": float(T.item()),
            "voltage": 0.0,
            "SOC": 0.0,
            "heat": predicted_heat_tensor.detach().cpu().tolist(),
            "target_heat": self.target_heat.detach().cpu().tolist(),
        }
        self._record(total_loss, metrics)
        return total_loss

# ───────────────────────── Evolution loop ──────────────────────────
def evolution(generations: int = 100, children_per_gen: int = 3):
    population = initialize_population()

    for gen in range(generations):
        print(f"\n[GA] Generation {gen}")
        tasks = []
        for c in range(children_per_gen):
            p1 = _tournament(population)
            p2 = _tournament(population)
            while p1["file"] == p2["file"]:
                p2 = _tournament(population)
            tasks.append((gen, c, str(p1["file"]), str(p2["file"])))

        with _MP_CTX.Pool() as pool:
            children_rows = [r for r in pool.map(_spawn_child, tasks) if r]

        _append_logs(children_rows)

        for row in children_rows:
            worst = max(population, key=lambda d: d["loss"])
            if row["Child_Loss"] < worst["loss"]:
                population.remove(worst)
                population.append({
                    "id": f"Gen{row['Generation']}_Child{row['Child_Number']}",
                    "folder": Path(row["Child_Folder"]),
                    "file": Path(row["Child_File"]),
                    "loss": row["Child_Loss"],
                })

        best = min(population, key=lambda d: d["loss"])
        print("[GA] Best so far →", best["id"], "loss", best["loss"])

# ────────────────────  Resume + Continue helpers  ─────────────────────
def _latest_loss(indiv_folder: Path, algo_file: Optional[Path] = None) -> float:
    """Return the most-recent loss from <folder>/results.csv (or 1e6)."""
    candidates = []
    if algo_file is not None:
        candidates.append(indiv_folder / f"results_{algo_file.stem}.csv")
    candidates.append(indiv_folder / "results.csv")
    for csv_path in candidates:
        if csv_path.exists():
            try:
                return float(pd.read_csv(csv_path).iloc[-1]["loss"])
            except Exception:
                continue
    return 1e6

def reconstruct_population(
    results_dir: Path = _RESULTS_DIR,
    gen_log_path: Path = _GEN_LOG,
) -> tuple[List[Dict], int]:
    """
    Rebuild the current 10-member population from files on disk and
    return (population, next_generation_index).
    """
    # 1) Original ten “Individual_*”
    population: List[Dict] = []
    for folder in results_dir.glob("Individual_*"):
        algo_file = folder / "algorithm.py"
        if algo_file.exists():
            population.append(
                {
                    "id": folder.name,
                    "folder": folder,
                    "file": algo_file,
                    "loss": _latest_loss(folder, algo_file),
                }
            )

    # 2) Replay generation_log.csv to reproduce replacements
    if gen_log_path.exists():
        df = (
            pd.read_csv(gen_log_path)
            .sort_values(["Generation", "Child_Number"])
            .reset_index(drop=True)
        )
        for _, row in df.iterrows():
            worst = max(population, key=lambda d: d["loss"])
            if row["Child_Loss"] < worst["loss"]:
                population.remove(worst)
                population.append(
                    {
                        "id": f"Gen{row.Generation}_Child{row.Child_Number}",
                        "folder": Path(row.Child_Folder),
                        "file": Path(row.Child_File),
                        "loss": row.Child_Loss,
                    }
                )
        next_gen = int(df["Generation"].max()) + 1
    else:
        next_gen = 0  # GA never started; only Individuals exist

    # keep exactly 10
    population = sorted(population, key=lambda d: d["loss"])[:10]
    return population, next_gen

def continue_evolution(
    population: List[Dict],
    start_gen: int,
    *,
    generations: int = 100,
    children_per_gen: int = 3,
):
    """
    Carry on evolving *population* for another *generations* generations.
    """
    for gen in range(start_gen, start_gen + generations):
        print(f"\n[GA] Generation {gen}")
        tasks = []
        for c in range(children_per_gen):
            p1 = _tournament(population)
            p2 = _tournament(population)
            while p1["file"] == p2["file"]:
                p2 = _tournament(population)
            tasks.append((gen, c, str(p1["file"]), str(p2["file"])))

        with _MP_CTX.Pool() as pool:
            children_rows = [r for r in pool.map(_spawn_child, tasks) if r]

        _append_logs(children_rows)

        for row in children_rows:
            worst = max(population, key=lambda d: d["loss"])
            if row["Child_Loss"] < worst["loss"]:
                population.remove(worst)
                population.append(
                    {
                        "id": f"Gen{row['Generation']}_Child{row['Child_Number']}",
                        "folder": Path(row["Child_Folder"]),
                        "file": Path(row["Child_File"]),
                        "loss": row["Child_Loss"],
                    }
                )

        best = min(population, key=lambda d: d["loss"])
        print("[GA] Best so far →", best["id"], "loss", best["loss"])

# ───────────────────────── Test function ──────────────────────────
def test_simulation():
    """Test the simulation with a known algorithm to verify it works correctly."""
    print("\n" + "="*60)
    print("Testing simulation with Individual_3 algorithm")
    print("="*60 + "\n")

    test_dir = _RESULTS_DIR / "test_simulation"
    test_dir.mkdir(parents=True, exist_ok=True)
    algo_path = test_dir / "algorithm.py"

    test_code = '''def current_function():
    Q = 1.7  # watts
    R0 = 0.02  # ohms
    t = pybamm.t
    # Multi-harmonic variation, kept positive by small amplitudes
    R_t = R0 * (1 + 0.15 * pybamm.sin(2 * 3.14 * t / 1000) + 0.10 * pybamm.cos(2 * 3.14 * t / 700))
    i_mag = pybamm.sqrt(Q / R_t)
    # Rational saturation to keep |i| <= 10 A
    i = -10 * (i_mag / (10 + i_mag))
    return i
'''

    try:
        print("1. Saving algorithm code...")
        func = _save_algo(test_code, algo_path)
        print("   ✓ Algorithm loaded successfully")

        print("\n2. Creating RunCharging instance...")
        runner = RunCharging(
            result_folder=test_dir,
            charging_current=func,
            target_heat_mode=TARGET_HEAT_MODE
        )
        print("   ✓ RunCharging created successfully")

        print("\n3. Running simulation...")
        loss = runner.run_simulation({})
        print(f"   ✓ Simulation completed! Loss: {loss:.6f}")

        print("\n4. Checking results...")
        results_csv = test_dir / "results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            print("   ✓ Results CSV created successfully")
            print("\n   Results summary:")
            print(df.to_string(index=False))
        else:
            print("   ✗ Results CSV not found")

        return loss

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        traceback.print_exc()
        return None

# ───────────────────────── Entry point ──────────────────────────
if __name__ == "__main__":
    # Optionally: test_simulation()
    # test_simulation()

    # Run full evolution with specified generations
    evolution(generations=50, children_per_gen=3)

    # Example to continue:
    # pop, next_gen = reconstruct_population()
    # continue_evolution(population=pop, start_gen=next_gen, generations=30, children_per_gen=3)

    # export $(grep -v '^#' .env | xargs)
    # python Control/llm_evoluation.py
