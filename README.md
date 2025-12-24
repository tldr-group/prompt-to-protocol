# Prompt-to-Protocol

A framework for optimizing battery charging protocols using neural networks and Bayesian optimization (SAASBO).

## Environment Setup

### Prerequisites

- Python 3.9+
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/prompt-to-protocol.git
cd prompt-to-protocol
```

2. Create a virtual environment:

```bash
conda create -n charging python=3.12.3
conda activate charging
```

3. Install dependencies:

**For Case 1 & Case 2 (Constant Heating Optimization):**

```bash
pip install -r requirements.txt
```

> Note: Case 1 & Case 2 require `pybamm==24.1`

**For Case 3 (Adaptive Charging Optimization):**

```bash
pip install ax-platform==0.4.0 deap==1.4.1 matplotlib==3.8.2 "numpy<2" openai==1.51.2 optuna==4.0.0 pandas==2.2.3 scipy==1.14.1 termcolor==2.5.0
pip install pybamm==25.1.1
```

> Note: Case 3 requires `pybamm==25.1.1` due to API changes in PyBaMM

## Quick Start: Running MWE (Minimum Working Example)

The `scripts/` directory contains self-contained examples that you can run directly:

### MWE Simulation

Test the battery simulation:

```bash
python scripts/MWE_simulation.py
```

### MWE Optimization

Run a complete SAASBO optimization:

```bash
python scripts/MWE_optimization.py
```

This will:
- Initialize a neural network for charging current control
- Run 30 Sobol initialization trials
- Run 60 batches of SAASBO optimization (3 samples per batch)
- Save results to `./MWE_results/`

## Project Structure

```
prompt-to-protocol/
├── scripts/                    # MWE scripts (standalone examples)
│   ├── MWE_optimization.py     # Complete optimization example
│   └── MWE_simulation.py       # Simulation example
├── src/
│   ├── p2o/                    # Prompt-to-Optimizer
│   │   ├── evaluate_model/     # Optimization methods
│   │   │   ├── SAABO_constant_heating.py   # Case 2
│   │   │   ├── SAABO_adaptive_RNN.py       # Case 3
│   │   │   ├── ECM_gradient_descent.py      # Case 1 
│   │   │   └── random_constant_heating.py   # Case 2
│   │   ├── simulation/         # Battery simulation
│   │   │   ├── sim_p2o_c1_c2.py  # Case 1 & 2 simulation
│   │   │   └── sim_p2o_c3.py     # Case 3 simulation
│   │   ├── evolution/          # Evolutionary algorithms
│   │   ├── evaluation/         # Evaluation scripts
│   │   └── llm_generation/     # LLM-based generation
│   ├── p2p/                    # Prompt-to-Protocol
│   │   ├── simulation/
│   │   ├── evolution/
│   │   └── llm_generation/
│   └── tools/                  # Utility functions
├── experiments/                # Experiment results
├── plots/                      # Visualization notebooks
├── requirements.txt
└── README.md
```

## Key Parameters

The SAASBO optimization uses the following default parameters:

- `N_INIT = 30`: Number of initial Sobol sampling points
- `BATCH_SIZE = 3`: Number of samples per SAASBO batch
- `N_BATCHES = 60`: Number of SAASBO optimization batches
- Search space: All neural network parameters bounded in `[-1.0, 1.0]`

## Output

After running optimization, results are saved to the specified folder:

- `optimization_results.csv`: All evaluated parameters and losses
- `optimal_sim_plot.png`: Visualization of the best simulation result
- `best_loss.txt`: The best loss value achieved

## License

See [LICENSE](LICENSE) for details.
