# Prompt-to-Protocol

From Prompt to Protocol: Fast Charging Batteries with Large Language Models

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

**For Case 1 & Case 2 (Constant / Predefined Heating Optimization):**

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

### API Key Configuration

For LLM-based optimization, you need to configure your OpenAI API key:

**Option 1: Using a `.env` file (Recommended)**

1. Create a `.env` file in the project root:

```bash
touch .env
```

2. Add your API key to the `.env` file:

```
export OPENAI_API_KEY="your-api-key-here"
```

3. Load the environment variables before running scripts:

```bash
source .env
```

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
- Save results to `./experiments/MWE_results/`

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

## License

See [LICENSE](LICENSE) for details.
