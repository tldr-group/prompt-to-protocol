import os
import re
import pandas as pd
from ac_nn_generation_multiprocessing import generate_initialization
from datetime import datetime
import traceback

def extract_save_multi_network_code(response, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[DEBUG] Saving network files to: {out_dir}")  # Add logging
    code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
    extracted_blocks = []
    for i, code_block in enumerate(code_blocks, start=1):
        code_block = code_block.strip()
        filename = f"network_{i}.py"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, 'w') as file:
            file.write(code_block)
        print(f"[DEBUG] Saved: {filepath}")  # Add logging
        extracted_blocks.append(filepath)
    print(f"[DEBUG] Total files saved: {len(extracted_blocks)}")  # Add logging
    return extracted_blocks

def initialize_pool_folder(continue_folder=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 确定仿真结果目录
    if continue_folder:
        simulation_folder = os.path.join(script_dir, continue_folder)
        os.makedirs(simulation_folder, exist_ok=True)
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        simulation_folder = os.path.join(script_dir, f"Simulation_results_LLM_{current_time}")
        os.makedirs(simulation_folder, exist_ok=True)

    pool_loss_path = os.path.join(simulation_folder, 'pool_best_loss.csv')
    pool_folder = os.path.join(script_dir, 'Pool_LLM')
    os.makedirs(pool_folder, exist_ok=True)

    if not os.path.exists(pool_loss_path):
        try:
            response = generate_initialization()
            # 直接用本文件里的函数，并指定输出目录为 Pool_LLM
            saved_files = extract_save_multi_network_code(response, pool_folder)

            # 如果生成的 code 少于 10 个，下面这块会指向不存在的文件；这里用实际生成的文件列表
            initial = [{'File': fpath, 'Best_Total_Loss': float('inf')} for fpath in saved_files]
            pd.DataFrame(initial).to_csv(pool_loss_path, index=False)
            print(f"Initialization complete. Saved {len(saved_files)} files to {pool_folder}")
        except Exception:
            traceback.print_exc()
    else:
        print("Initialization already done.")

    return pool_loss_path, pool_folder, simulation_folder

if __name__ == "__main__":
    initialize_pool_folder()