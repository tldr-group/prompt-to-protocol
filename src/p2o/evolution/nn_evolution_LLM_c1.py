import shutil
import sys
import pandas as pd
from nn_evaluation import evaluate_models_in_parallel
from nn_generation import generate_new_network
from nn_generation_multiprocessing import generate_initialization
import numpy as np
from utils import *
import os
import random
import traceback
import pandas as pd
from datetime import datetime
import traceback
from scipy.stats import entropy  # Import entropy from scipy

# run for 200 iterations, need Pool_LLM folder

def select_parents(pool_loss_file='Pool_LLM/pool_best_loss.csv', tournament_size=3):
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
    # """
    # Choose the appropriate network generation function based on parents' loss and pool entropy.
    
    # Args:
    #     parent1_loss (float): Loss of the first parent.
    #     parent2_loss (float): Loss of the second parent.
    #     pool_entropy (float): Entropy of the pool's loss distribution.
        
    # Returns:
    #     function: The selected network generation function.
    # """
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



import os
import pandas as pd
from datetime import datetime
from nn_evaluation import evaluate_models_in_parallel  
import shutil

import os
import pandas as pd
from datetime import datetime
import sys

def initialize_pool_and_simulation_folders(iteration_index):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a Simulation_results folder with a timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    simulation_folder = os.path.join(
        script_dir,
        f"Simulation_results_LLM_iter{iteration_index + 1:03d}_{current_time}"
    )
    os.makedirs(simulation_folder, exist_ok=True)

    # Define the path for pool_best_loss.csv
    pool_loss_path = os.path.join(simulation_folder, 'pool_best_loss.csv')

    # Define the network files to evaluate, only evaluating network_1.py to network_10.py
    pool_folder = os.path.join(script_dir, 'Pool_LLM')
    if os.path.exists(pool_folder):
        try:
            shutil.rmtree(pool_folder)
        except Exception as e:
            print(f"Error clearing existing pool folder {pool_folder}: {e}")
            raise
    os.makedirs(pool_folder, exist_ok=True)

    # Ensure the freshly generated Pool_LLM modules take precedence when imported
    pool_folder = os.path.abspath(pool_folder)
    while pool_folder in sys.path:
        sys.path.remove(pool_folder)
    sys.path.insert(0, pool_folder)
    clear_cache()

    network_files = [f"network_{i}.py" for i in range(1, 11)]

    print(f"初始化新的 pool，迭代 {iteration_index + 1}")

    try:
        response = generate_initialization()
        print("生成网络代码成功。")

        # 提取并保存网络代码到 Pool_LLM 文件夹
        extract_save_multi_network_code(response, output_dir=pool_folder)
        print("已保存网络代码到 Pool_LLM 文件夹。")

    except Exception as e:
        print(f"初始化 pool 过程中出错: {e}")
        traceback.print_exc()
        raise

    # Create a DataFrame to store network files and initial loss values
    initial_results = []
    for network_file in network_files:
        network_path = os.path.join(pool_folder, network_file)
        initial_results.append({'File': network_path, 'Best_Total_Loss': float('inf')})

    df_initial = pd.DataFrame(initial_results)
    df_initial.to_csv(pool_loss_path, index=False)
    print(f"Initialization complete: saved to {pool_loss_path}")

    return pool_loss_path, pool_folder, simulation_folder


def evaluate_undefined_models(pool_loss_path, pool_folder, simulation_folder):
    # 读取 pool_best_loss.csv 文件
    df = pd.read_csv(pool_loss_path)
    clear_cache()
    
    # 查找 loss 值为 NaN 或 inf 的网络，意味着它们尚未被评估
    undefined_loss = df['Best_Total_Loss'].isnull() | (df['Best_Total_Loss'] == float('inf'))

    models_to_evaluate = df[undefined_loss]['File'].tolist()

    if models_to_evaluate:
        print(f"需要评估 {len(models_to_evaluate)} 个网络文件.")
        # 获取没有扩展名的模块名称
        module_names = [os.path.splitext(os.path.basename(f))[0] for f in models_to_evaluate]

        for original_file, module_name in zip(models_to_evaluate, module_names):
            try:
                new_result_folders, new_nn_loss = evaluate_models_in_parallel(
                    module_names=[module_name],
                    main_folder=simulation_folder,
                    evaluate_model_module='evaluate_model.ECM_gradient_descent',
                    num_workers=1
                )
            except Exception as e:
                print(f"在评估网络 {original_file} 时发生错误: {e}")
                traceback.print_exc()
                idx = df.index[df['File'] == original_file].tolist()
                if idx:
                    df = df.drop(idx[0]).reset_index(drop=True)
                    print(f"已从池中移除出错的文件: {original_file}")
                if os.path.exists(original_file):
                    try:
                        os.remove(original_file)
                    except Exception as rm_error:
                        print(f"删除出错文件 {original_file} 时失败: {rm_error}")
                continue

            if not new_result_folders or not new_nn_loss:
                print(f"评估 {original_file} 未返回有效结果，将跳过。")
                continue

            new_folder = new_result_folders[0]
            loss = new_nn_loss[0]
            new_file_path = os.path.join(new_folder, "model_architecture.py")

            if os.path.exists(new_file_path):
                print(f"更新文件路径: {new_file_path}，对应的损失值: {loss}")
                idx = df.index[df['File'] == original_file].tolist()
                if idx:
                    df.loc[idx[0], 'File'] = new_file_path
                    df.loc[idx[0], 'Best_Total_Loss'] = loss
                    print(f"文件 {new_file_path} 的损失值成功更新为 {loss}")
                else:
                    print(f"文件路径 {original_file} 在 DataFrame 中找不到！")
            else:
                print(f"评估结果文件 {new_file_path} 不存在，将跳过。")

        df.to_csv(pool_loss_path, index=False)
        print(f"评估完成，并更新 {pool_loss_path} 中的 loss 值.")
    else:
        print("没有需要评估的 loss 值，所有网络均已评估.")
    
    # 返回更新后的 DataFrame
    return df



def main():
    # Set number of iterations to 10
    num_iterations = 120
    top_n = 10  # Retain the top_n models in the pool

    # Initialize pool and simulation folders once before iterations
    pool_loss_path, pool_folder, simulation_folder = initialize_pool_and_simulation_folders(0)
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        # Evaluate all models with NaN or inf loss values, and get the updated DataFrame
        df = evaluate_undefined_models(pool_loss_path, pool_folder, simulation_folder)
    
        new_entries = []  # List to hold new models after they are evaluated
        module_names = []  # Store the names of generated modules
    
        # Calculate the current pool's entropy
        pool_losses = df['Best_Total_Loss'].tolist()
        current_entropy = calculate_loss_entropy(pool_losses)
        print(f"Current pool loss entropy: {current_entropy:.4f}")
    
        for child_num in range(1, 4):
            print(f"\n--- Generating Child Network {child_num} ---")
            # Select new parents for each child network
            try:
                parent1, parent2 = select_parents(pool_loss_file=pool_loss_path, tournament_size=3)
                print(f"Selected parents for child {child_num}:\n 1. {parent1['File']} (Loss: {parent1['Best_Total_Loss']})\n 2. {parent2['File']} (Loss: {parent2['Best_Total_Loss']})")
                # clear_cache()
            except Exception as e:
                print(f"Error selecting parents: {e}")
                traceback.print_exc()
                continue
    
            # Read parent network codes
            try:
                with open(parent1['File'], 'r') as f:
                    seed1_code = f.read()
                with open(parent2['File'], 'r') as f:
                    seed2_code = f.read()
            except Exception as e:
                print(f"Error reading parent files: {e}")
                traceback.print_exc()
                continue
    
            # Choose the appropriate generation function
            try:
                generation_func = choose_generation_function(parent1['Best_Total_Loss'], parent2['Best_Total_Loss'], current_entropy)
            except Exception as e:
                print(f"Error choosing generation function: {e}")
                traceback.print_exc()
                continue
    
            # Generate child network code using the chosen generation function
            try:
                new_code = generation_func(seed1_code, seed2_code)
                if not isinstance(new_code, str):
                    print(f"Generated code for child {child_num} is not a string. Skipping.")
                    continue
                
                print(f"\n--- Generated Code for Child {child_num} (Iteration {iteration + 1}) ---\n{new_code}\n--- End of Code ---\n")

                # Save new code in the simulation folder but do not add it to the CSV yet
                child_filename = f"child_network_{child_num}.py"
                new_file_path = extract_save_new_network_code(new_code, child_filename)
                module_names.append(os.path.splitext(child_filename)[0])  # Add the generated network name to be evaluated
                
            except Exception as e:
                print(f"Error generating code for child {child_num}: {e}")
                traceback.print_exc()
                continue
        
        clear_cache()
        # Evaluate all generated child networks after generation
        if module_names:
            try:
                # Perform parallel evaluation
                new_result_folders, new_nn_loss = evaluate_models_in_parallel(
                    module_names=module_names,
                    main_folder=simulation_folder,
                    evaluate_model_module='evaluate_model.ECM_gradient_descent',
                    num_workers=len(module_names)
                )
            except Exception as e:
                print(f"Error evaluating models: {e}")
                # If evaluation fails, remove result folders
                for folder in new_result_folders:
                    if os.path.exists(folder):
                        try:
                            shutil.rmtree(folder)
                            print(f"Removed folder {folder} due to evaluation error.")
                        except Exception as e_rm:
                            print(f"Error removing folder {folder}: {e_rm}")
                continue
    
            # Update the pool with evaluated networks and their loss values
            for new_loss, new_folder in zip(new_nn_loss, new_result_folders):
                try:
                    new_code_path = os.path.join(new_folder, "model_architecture.py")
                    if os.path.exists(new_code_path):
                        new_file = os.path.join(pool_folder, f"{os.path.basename(new_folder)}.py")
                        shutil.copy(new_code_path, new_file)  # Save evaluated result
                        
                        # Add the evaluated result to the pool with its actual loss
                        new_entries.append({'File': new_file, 'Best_Total_Loss': new_loss})
                        print(f"New file {new_file} added to the pool with loss {new_loss}.")
                except Exception as e:
                    print(f"Error processing evaluation result {new_folder}: {e}")
                    traceback.print_exc()
                    continue
    
        # Save new entries into the pool_best_loss.csv only after evaluation
        if new_entries:
            try:
                df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)
                df = df.sort_values(by='Best_Total_Loss').reset_index(drop=True)
                
                # Keep only the top_n models and remove others from the pool
                if len(df) > top_n:
                    to_remove = df.iloc[top_n:]['File'].tolist()
                    df = df.iloc[:top_n]
                    for file in to_remove:
                        if os.path.exists(file):
                            try:
                                os.remove(file)
                                print(f"Removed file {file} from pool.")
                            except Exception as e_rm:
                                print(f"Error removing file {file}: {e_rm}")
                        else:
                            print(f"File {file} not found for removal.")
                
                # Save the updated pool
                df.to_csv(pool_loss_path, index=False)
                print(f"Pool updated. Retained top {top_n} models.")
            except Exception as e:
                print(f"Error updating the pool: {e}")
                traceback.print_exc()
        else:
            print("No new valid entries added to the pool in this iteration.")
    
    print("\nEvolution process completed.")



if __name__ == "__main__":
    success_count = 0  
    max_successes = 10  

    for i in range(1):  
        if success_count >= max_successes:
            print(f"\n=== Successfully ran {success_count} iterations. Stopping. ===")
            break  

        print(f"\n=== Running Main Iteration {i+1} ===")
        
        try:
            main()  #
            success_count += 1  
        except Exception as e:
            print(f"Error during iteration {i+1}: {e}")
            print("Skipping to the next iteration...")

    print(f"\n=== Completed with {success_count} successful iterations. ===")
