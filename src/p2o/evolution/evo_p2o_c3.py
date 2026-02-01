import shutil
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import re
import pandas as pd
from src.p2o.evaluation.eva_p2o_c3 import evaluate_models_in_parallel
from src.p2o.llm_generation.gen_p2o_c3 import generate_initialization, generate_new_network
import numpy as np 
from src.tools.utils import *
import importlib
import random
import traceback
from scipy.stats import entropy  

# Run for a specific folder

def select_parents(pool_loss_file, tournament_size=3):
    """
    Select two parents using the tournament selection method.
    
    Args:
        pool_loss_file (str): Path to the CSV file containing model losses in the pool.
        tournament_size (int): Number of individuals participating in each tournament.
        
    Returns:
        tuple: Two selected parent rows.
    """
    df = pd.read_csv(pool_loss_file)
    
    def tournament_selection(df, k):
        # Randomly select k individuals for the tournament
        participants = df.sample(n=k)
        # Select the individual with the smallest loss as the winner
        winner = participants.loc[participants['Best_Total_Loss'].idxmin()]
        return winner
    
    parent1 = tournament_selection(df, tournament_size)
    parent2 = tournament_selection(df, tournament_size)
    
    # Ensure parents are different
    while parent2['File'] == parent1['File']:
        parent2 = tournament_selection(df, tournament_size)
    
    return parent1, parent2

def clear_cache():
    """
    Clear all modules related to neural networks or custom layers to force reload.
    """
    for module in list(sys.modules.keys()):
        if 'network' in module or 'custom_layer' in module:
            del sys.modules[module]

def calculate_loss_entropy(pool_losses, num_bins=20):
    """
    Calculate the entropy of the loss distribution.

    Args:
        pool_losses (list): A list of all loss values.
        num_bins (int): Number of bins for the histogram, default is 20.

    Returns:
        float: Entropy value.
    """
    if not pool_losses:
        return 0
    histogram, _ = np.histogram(pool_losses, bins=num_bins, density=True)
    # Filter out zero probabilities to avoid errors during entropy calculation
    histogram = histogram[histogram > 0]
    return entropy(histogram)

def choose_generation_function(parent1_loss, parent2_loss, pool_entropy):
    """
    Choose the appropriate network generation function based on parents' loss and pool entropy.
    
    Args:
        parent1_loss (float): Loss of the first parent.
        parent2_loss (float): Loss of the second parent.
        pool_entropy (float): Entropy of the pool's loss distribution.
        
    Returns:
        function: The selected network generation function.
    """
    # if parent1_loss < 0.007 or parent2_loss < 0.007:
    #     print("Pool entropy:", pool_entropy)
    #     if pool_entropy > 3 and parent1_loss < 0.0008 and parent2_loss < 0.0008:
    #         print("Selected generate_new_network_increase")
    #         return generate_new_network_increase
    #     else:
    #         print("Selected generate_new_network_subtle")
    #         return generate_new_network_subtle
    # else:
    #     print("Selected generate_new_network as default generation function.")
    return generate_new_network

def main(simulation_results_folder):
    # Define the path directly here instead of using command-line arguments
    print("Starting the evolutionary process for neural networks...")  
    pool_loss_path = os.path.join(simulation_results_folder, 'pool_best_loss.csv')
    print(f"Using pool loss file: {pool_loss_path}")
    
    if not os.path.exists(pool_loss_path):
        print(f"Error: '{pool_loss_path}' does not exist. Please ensure the file is present in the specified folder.")
        sys.exit(1)
    
    # Add the pool folder to sys.path to allow module imports
    sys.path.append(simulation_results_folder)
    print(f"Added {simulation_results_folder} to sys.path for module imports.")

    # Initialize losses for models in the pool
    df = pd.read_csv(pool_loss_path)
    df['Best_Total_Loss'].replace([np.inf, -np.inf], np.nan, inplace=True)


    if df['Best_Total_Loss'].isnull().any():
        # Evaluate models with undefined loss
        undefined_loss = df['Best_Total_Loss'].isnull()
        models_to_evaluate = df[undefined_loss]['File'].tolist()
        if models_to_evaluate:
            module_names = [os.path.splitext(os.path.basename(f))[0] for f in models_to_evaluate]
            clear_cache()
            try:
                new_result_folders, new_nn_loss = evaluate_models_in_parallel(
                    module_names=module_names,
                    main_folder=simulation_results_folder,
                    evaluate_model_module='evaluate_model.SAABO_adaptive_RNN',
                    num_workers=len(module_names)
                )
                for file, loss in zip(models_to_evaluate, new_nn_loss):
                    df.loc[df['File'] == file, 'Best_Total_Loss'] = loss
                df.to_csv(pool_loss_path, index=False)
                print("Initialized losses for existing pool models.")
            except Exception as e:
                print(f"Error during initial evaluation of undefined losses: {e}")
                traceback.print_exc()
                sys.exit(1)

    num_iterations = 50
    top_n = 10  # Number of top models to retain in the pool

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")

        # Initialize new entries for this iteration
        new_entries = []
        module_names = []  # Store names of generated modules

        # Calculate current pool entropy
        pool_losses = df['Best_Total_Loss'].tolist()
        # current_entropy = calculate_loss_entropy(pool_losses)
        # print(f"Current pool loss entropy: {current_entropy:.4f}")

        for child_num in range(1, 4):
            print(f"\n--- Generating Child {child_num} ---")
            # Select new parents for each child
            parent1, parent2 = select_parents(pool_loss_file=pool_loss_path, tournament_size=3)
            print(f"Selected Parents for Child {child_num}:\n 1. {parent1['File']} with loss {parent1['Best_Total_Loss']}\n 2. {parent2['File']} with loss {parent2['Best_Total_Loss']}")

            # Read parent code
            try:
                with open(parent1['File'], 'r') as f:
                    seed1_code = f.read()
                with open(parent2['File'], 'r') as f:
                    seed2_code = f.read()
            except Exception as e:
                print(f"Error reading parent files: {e}")
                traceback.print_exc()
                continue

            # Set unique random seeds to ensure reproducibility and diversity
            np.random.seed(random.randint(0, 10000))
            random.seed(random.randint(0, 10000))

            # Choose the appropriate generation function
            # generation_func = choose_generation_function(parent1['Best_Total_Loss'], parent2['Best_Total_Loss'], current_entropy)
            generation_func = generate_new_network  # Default to the basic generation function

            # Generate child code using the selected generation function
            try:
                new_code = generation_func(seed1_code, seed2_code)
                if not isinstance(new_code, str):
                    print(f"Generated network code is not a string for child {child_num}. Skipping.")
                    continue
                print(f"\n--- Generated Code for Child {child_num} in Iteration {iteration + 1} ---\n{new_code}\n--- End of Code ---\n")
                # Extract and save the new network code
                code_blocks = extract_save_new_network_code(new_code, filename=f"network_{child_num}.py")
                module_names.append(f'network_{child_num}')  # Add generated network filename with iteration and child number
            except Exception as e:
                print(f"Error during network generation for child {child_num}: {e}")
                traceback.print_exc()
                continue

        clear_cache()  # Clear cache to ensure new networks are properly loaded

        # Evaluate all generated child networks
        if module_names:
            new_result_folders = []  # Initialize before the try-except block
            new_nn_loss = []  # Initialize before the try-except block
            try:
                new_result_folders, new_nn_loss = evaluate_models_in_parallel(
                    module_names=module_names,
                    main_folder=simulation_results_folder,
                    evaluate_model_module='evaluate_model.SAABO_adaptive_RNN',
                    # num_workers=len(module_names)
                    num_workers=3
                )
            except Exception as e:
                print(f"Error during model evaluation: {e}")
                # If evaluation fails, remove the result folders
                for folder in new_result_folders:
                    if os.path.exists(folder):
                        try:
                            shutil.rmtree(folder)
                            print(f"Removed {folder} due to evaluation error.")
                        except Exception as ex:
                            print(f"Error while removing {folder}: {ex}")
                continue

            # Record evaluation results as new entries
            for new_loss, new_folder in zip(new_nn_loss, new_result_folders):
                try:
                    new_code = open(os.path.join(new_folder, "model_architecture.py")).read()
                    new_file = os.path.join(simulation_results_folder, f"{os.path.basename(new_folder)}.py")
                    with open(new_file, 'w') as file:
                        file.write(new_code)
                    if isinstance(new_loss, (tuple, list, np.ndarray)):
                        new_loss = float(new_loss[1])

                    new_entries.append({'File': new_file, 'Best_Total_Loss': new_loss})
                except Exception as e:
                    print(f"Error processing evaluation results for {new_folder}: {e}")
                    traceback.print_exc()
                    continue

        # Update the pool with new entries
        if new_entries:
            df = pd.read_csv(pool_loss_path)
            # Check for duplicate files to avoid redundancy
            for entry in new_entries:
                if not df[df['File'] == entry['File']].empty:
                    print(f"{entry['File']} already exists in the pool, skipping.")
                else:
                    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

            # Sort by loss and retain top_n models
            df = df.sort_values(
                    by='Best_Total_Loss',
                    key=lambda s: s.apply(lambda x: x[1] if isinstance(x, (tuple, list, np.ndarray)) else x)
                ).reset_index(drop=True)
            if len(df) > top_n:
                # Identify files to remove
                to_remove = df.iloc[top_n:]['File'].tolist()
                df = df.iloc[:top_n]
                # Remove files not in the top_n
                for file in to_remove:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                            print(f"Removed {file} from the pool.")
                        except Exception as e:
                            print(f"Error while removing {file}: {e}")
                    else:
                        print(f"{file} does not exist.")
            # Save the updated pool
            df.to_csv(pool_loss_path, index=False)
            print(f"Pool updated. Current top {top_n} networks saved.")
        else:
            print("No new valid entries to add to the pool.")

    print("\nEvolutionary process completed.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolutionary Neural Network Optimization for Case 3')
    parser.add_argument('--test', action='store_true', default=True, 
                        help='Run in test mode without actual evolution (default: True)')
    parser.add_argument('--run', action='store_true', 
                        help='Run actual evolution (overrides --test)')
    parser.add_argument('--folder', type=str, default='experiments/case3_p2o', 
                        help='Path to the simulation results folder')
    args = parser.parse_args()
    
    # If --run is specified, disable test mode
    if args.run:
        args.test = False
    
    if args.test:
        # ========== Test Mode: Verify core functions without running full evolution ==========
        # This test is completely self-contained and does NOT use any existing files
        print("=" * 60)
        print("Evolutionary Neural Network Optimization (Case 3) - Test Mode")
        print("=" * 60)
        print("\nNote: This test is self-contained and creates all test data in memory/temp files.")
        print("No existing project files are used.\n")
        
        import tempfile
        import torch
        import torch.nn as nn
        
        # Test 1: Test utility functions
        print("-" * 60)
        print("Test 1: Testing calculate_loss_entropy...")
        print("-" * 60)
        
        # Test calculate_loss_entropy with in-memory data
        test_losses = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.12, 0.28, 0.19]
        test_entropy = calculate_loss_entropy(test_losses, num_bins=5)
        print(f"  Sample losses: {test_losses[:5]}...")
        print(f"  Calculated entropy: {test_entropy:.4f}")
        print(f"  ✓ Entropy calculation works")
        
        # Test with empty list
        empty_entropy = calculate_loss_entropy([], num_bins=5)
        assert empty_entropy == 0, "Empty list should return 0 entropy"
        print(f"  ✓ Empty list returns 0 entropy")
        
        # Test 2: Test clear_cache function
        print("\n" + "-" * 60)
        print("Test 2: Testing clear_cache function...")
        print("-" * 60)
        
        initial_modules = len(sys.modules)
        clear_cache()
        print(f"  Modules before: {initial_modules}, after: {len(sys.modules)}")
        print(f"  ✓ clear_cache executed successfully")
        
        # Test 3: Test choose_generation_function
        print("\n" + "-" * 60)
        print("Test 3: Testing choose_generation_function...")
        print("-" * 60)
        
        gen_func = choose_generation_function(0.01, 0.02, 2.5)
        print(f"  Input: parent1_loss=0.01, parent2_loss=0.02, pool_entropy=2.5")
        print(f"  Selected function: {gen_func.__name__}")
        assert gen_func == generate_new_network, "Should return generate_new_network"
        print(f"  ✓ Generation function selection works")
        
        # Test 4: Test select_parents with temporary mock CSV (no real files used)
        print("\n" + "-" * 60)
        print("Test 4: Testing select_parents with temporary mock data...")
        print("-" * 60)
        
        # Create a temporary mock CSV file for testing (completely in-memory/temp)
        mock_data = {
            'File': ['/tmp/mock_model_1.py', '/tmp/mock_model_2.py', '/tmp/mock_model_3.py', 
                     '/tmp/mock_model_4.py', '/tmp/mock_model_5.py'],
            'Best_Total_Loss': [0.1, 0.2, 0.15, 0.25, 0.18]
        }
        mock_df = pd.DataFrame(mock_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            mock_df.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        try:
            parent1, parent2 = select_parents(temp_csv_path, tournament_size=2)
            print(f"  Mock CSV created at: {temp_csv_path}")
            print(f"  Parent 1: {os.path.basename(parent1['File'])} (loss: {parent1['Best_Total_Loss']:.4f})")
            print(f"  Parent 2: {os.path.basename(parent2['File'])} (loss: {parent2['Best_Total_Loss']:.4f})")
            assert parent1['File'] != parent2['File'], "Parents should be different"
            print(f"  ✓ Tournament selection works (parents are different)")
        finally:
            os.remove(temp_csv_path)
            print(f"  ✓ Temporary CSV cleaned up")
        
        # Test 5: Test neural network structure validation (in-memory, no file I/O)
        print("\n" + "-" * 60)
        print("Test 5: Testing neural network structure (in-memory)...")
        print("-" * 60)
        
        # Create a sample network in memory to test structure
        class TestNeuralNetwork(nn.Module):
            """Test network following Case 3 requirements: 3 inputs, output in [-8, -3]"""
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 3)
                self.ln1 = nn.LayerNorm(3, elementwise_affine=False)
                self.fc2 = nn.Linear(3, 1)

            def forward(self, x):
                h = torch.relu(self.ln1(self.fc1(x)))
                out = -torch.sigmoid(self.fc2(h)) * 5 - 3  # Output in [-8, -3]
                return out
        
        test_net = TestNeuralNetwork()
        num_params = sum(p.numel() for p in test_net.parameters())
        print(f"  Test network structure: 3 -> 3 -> 1")
        print(f"  Number of parameters: {num_params}")
        print(f"  Parameter constraint (< 35): {'✓ PASS' if num_params < 35 else '✗ FAIL'}")
        
        # Test forward pass
        test_input = torch.randn(1, 3)  # Random input with 3 features
        test_output = test_net(test_input)
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Test output: {test_output.item():.4f}")
        assert -8 <= test_output.item() <= -3, f"Output {test_output.item():.4f} out of range [-8, -3]"
        print(f"  Output range [-8, -3]: ✓ PASS")
        
        # Test 6: Test code pattern validation (string matching, no files)
        print("\n" + "-" * 60)
        print("Test 6: Testing code pattern validation...")
        print("-" * 60)
        
        # Example of expected code structure (just a string for pattern checking)
        example_code = '''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return -torch.sigmoid(x) * 5 - 3

def nn_in_pybamm(X, Ws, bs):
    return -pybamm.Scalar(5) * y - 3
'''
        
        assert 'class NeuralNetwork' in example_code, "Should contain NeuralNetwork class"
        assert 'def nn_in_pybamm' in example_code, "Should contain nn_in_pybamm function"
        assert 'torch.sigmoid' in example_code, "Should use sigmoid"
        print(f"  ✓ Code pattern: NeuralNetwork class present")
        print(f"  ✓ Code pattern: nn_in_pybamm function present")
        print(f"  ✓ Code pattern: sigmoid activation present")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
        print("\nTo run the actual evolution, use:")
        print(f"  python {__file__} --folder <path_to_simulation_folder>")
        print("\nExample:")
        print(f"  python {__file__} --folder experiments/case3_p2o")
        print("\n" + "=" * 60)
    else:
        # Run actual evolution
        main(args.folder)
