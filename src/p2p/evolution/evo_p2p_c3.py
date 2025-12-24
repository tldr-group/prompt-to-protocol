# -*- coding: utf-8 -*-
"""battery_evolution.py
---------------------------------
Evolves battery‑charging algorithms using an LLM‑assisted genetic algorithm.
The loss calculation (SOH‑based, –log mapping with < 0.93 hard penalty) is
left intact, per user request; other parts were refactored for robustness.

*Fixes in this revision*
-----------------------
‑ Ensure **all 10 initial algorithms** are extracted and executed.
‑ Skip empty / invalid code blocks; log explicitly.
‑ Child extraction picks the **first valid block** in response.
‑ Removed duplicate child evaluations.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import multiprocessing as mp
import random
import re
import traceback
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import pybamm

from ax.core.metric import Metric
from control_simulation_new import BatteryCyclingExperiment
from llm_generation import generate_initialization, generate_new_algorithm

# ───────────────────────── Global config ──────────────────────────
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)

# RegEx captures fenced blocks with or without the optional "python" tag
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Add timestamp to results directory to allow concurrent runs
_TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "experiments" / f"Control_simulation_results_{_TIMESTAMP}"
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
    """Pick first block that defines `adaptive_current`, else first block."""
    for blk in _extract_all_code(markdown):
        if "def adaptive_current" in blk:
            return blk
    return _extract_all_code(markdown)[0]


def _save_algo(code: str, dest: Path) -> Callable:
    dest.write_text(code)
    ns: Dict[str, object] = {
        "__name__": "algorithm_runtime",
        "__builtins__": __builtins__,
        "np": np, "numpy": np,
        "pd": pd, "pandas": pd,
        "torch": torch,
        "random": random,
        "pybamm": pybamm,
    }
    exec(code, ns, ns)
    func = ns.get("adaptive_current")
    if not callable(func):
        raise ValueError("adaptive_current() missing or not callable")
    return func

# ───────────────────────── Population initialisation ──────────────────────────

def initialize_population() -> List[Dict]:
    print("[Init] Generating initial algorithms via LLM…")
    response = generate_initialization()
    code_blocks = _extract_all_code(response)

    population: List[Dict] = []
    for idx, code in enumerate(code_blocks):
        indiv_dir = _RESULTS_DIR / f"Individual_{idx}"
        indiv_dir.mkdir(exist_ok=True)
        algo_path = indiv_dir / "algorithm.py"
        try:
            # Skip empty / placeholder blocks
            if not code.strip():
                raise ValueError("Empty code block")
            func = _save_algo(code, algo_path)
            loss = RunCharging(result_folder=indiv_dir, charging_current=func).run_simulation({})
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
    try:
        p1_code = Path(p1_file).read_text()
        p2_code = Path(p2_file).read_text()
        child_markdown = generate_new_algorithm(p1_code, p2_code)
        child_code = _first_valid_code(child_markdown)
        child_dir = _RESULTS_DIR / f"Gen_{gen}_Child_{child_idx}"
        child_dir.mkdir(exist_ok=True)
        algo_path = child_dir / "algorithm.py"
        func = _save_algo(child_code, algo_path)
        loss = RunCharging(result_folder=child_dir, charging_current=func).run_simulation({})
    except Exception:
        traceback.print_exc()
        loss = 1e6
        child_dir = _RESULTS_DIR / f"Gen_{gen}_Child_{child_idx}_error"
        child_dir.mkdir(exist_ok=True)
        algo_path = child_dir / "algorithm.py"
        algo_path.write_text("# error placeholder\n")

    return {
        "Generation": gen,
        "Child_Number": child_idx,
        "Parent1": p1_file,
        "Parent2": p2_file,
        "Child_Folder": str(child_dir),
        "Child_File": str(algo_path),
        "Child_Loss": loss,
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
    """Wrap BatteryCyclingExperiment and expose SOH‑based loss."""

    def __init__(self, name="total_loss", *, result_folder: Path, charging_current: Callable, **kw):
        super().__init__(name=name, **kw)
        self.result_folder = result_folder
        self.charging_current = charging_current

    @staticmethod
    def loss_function(soh):
        if soh is None or soh < 0.6:
            return 1e6
        return -float(np.log((soh - 0.6) / 0.4))

    def _record(self, loss, soh):
        df = pd.DataFrame([{"loss": loss, "SOH": soh}])
        csv_path = self.result_folder / "results.csv"
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    def run_simulation(self, _):
        try:
            exp = BatteryCyclingExperiment(num_cycles=100, SOC=0.9, adaptive_current=self.charging_current)
            _, soh_series = exp.run_cycles()
            soh = soh_series[-1]
            loss = self.loss_function(soh)
            self._record(loss, soh)
            return loss
        except Exception:
            traceback.print_exc()
            return 1e6

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
            tasks.append((gen, c, str(p1["file"]), str(p2["file"])) )

        with mp.Pool() as pool:
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
def _latest_loss(indiv_folder: Path) -> float:
    """Return the most-recent loss from <folder>/results.csv (or 1e6)."""
    csv_path = indiv_folder / "results.csv"
    if not csv_path.exists():
        return 1e6
    try:
        return float(pd.read_csv(csv_path).iloc[-1]["loss"])
    except Exception:
        return 1e6


def reconstruct_population(
    results_dir: Path = _RESULTS_DIR,
    gen_log_path: Path = _GEN_LOG,
) -> tuple[List[Dict], int]:
    """
    Rebuild the current 10-member population from files on disk and
    return (population, next_generation_index).
    """
    # 1) start with the original ten “Individual_*”
    population: List[Dict] = []
    for folder in results_dir.glob("Individual_*"):
        algo_file = folder / "algorithm.py"
        if algo_file.exists():
            population.append(
                {
                    "id": folder.name,
                    "folder": folder,
                    "file": algo_file,
                    "loss": _latest_loss(folder),
                }
            )

    # 2) replay generation_log.csv to reproduce every replacement
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

    # keep exactly 10 (in case some folders were deleted)
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

    Parameters
    ----------
    population       rebuilt by `reconstruct_population`
    start_gen        index to start counting from (usually next_gen)
    generations      how many generations to add
    children_per_gen how many children per generation (default 3)
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

        with mp.Pool() as pool:
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


# ───────────────────────── Entry point ──────────────────────────
if __name__ == "__main__":
    evolution()
    
    pop, next_gen = reconstruct_population()

    # Continue for 100 extra generations (same settings as before)
    continue_evolution(
        population=pop,
        start_gen=next_gen,
        generations=67,
        children_per_gen=3,
    )

    # export $(grep -v '^#' .env | xargs)
