import os
import itertools
from numpy import sort
from sklearn.neighbors import sort_graph_by_row_values
from gputracker.gputracker import get_logger, DispatchThread

gpus = list(range(8))

# Set the MKL threading layer to 'gnu'
os.environ['MKL_THREADING_LAYER'] = 'gnu'

# GPUs to use
# gpus = [5, 6, 7]

# Parameters to vary
num_layers_list = [14]  # You can add more values as needed
max_steps_list = [22500]  # You can add more values as needed
sortby_list = ["block_random"]  # You can add more values as needed
seed_list = [42, 357]  # Example seed values
model_name_or_path = 'meta-llama/Meta-Llama-3-8B'
cachedir = "/scratch/kinshuk/cache"  # Set your cache directory

# Logger setup
logger = get_logger('log', 'schedule_subspace.log')

# BASH command list
BASH_COMMAND_LIST = []

# Generate commands for each combination of parameters
for num_layers, max_steps, sortby, seed in itertools.product(num_layers_list, max_steps_list, sortby_list, seed_list):
    save_path = f"./results/{model_name_or_path}/seed_{seed}"

    cmd = (
        "OMP_NUM_THREADS=1 python /jumbo/yaoqingyang/kinshuk/LlaMAft/selft.py "
        f"--seed {seed} "
        f"--data_seed 7 "
        f"--output_dir {save_path} "
        f"--dataset alpaca "
        f"--max_eval_samples 50 "
        f"--dataloader_num_workers 1 "
        f"--do_eval false "
        f"--max_steps {max_steps} "
        f"--sortby {sortby} "
        f"--num_layers {num_layers} "
        f"--source_max_len 512 "
        f"--target_max_len 256 "
        f"--learning_rate 2e-6 "
        f"--per_device_train_batch_size 1 "
        f"--memlog "
        f"--cache_dir {cachedir} "
        f"--model_name_or_path {model_name_or_path} "
    )

    BASH_COMMAND_LIST.append(cmd)

# Dispatch thread setup
dispatch_thread = DispatchThread(
    "Alpaca dataset training",
    BASH_COMMAND_LIST,
    logger,
    gpu_m_th=500,
    gpu_list=gpus,
    maxcheck=0,
)

# Start and join the dispatch thread
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")