import os
import itertools
from numpy import sort
from sklearn.neighbors import sort_graph_by_row_values
from gputracker.gputracker import get_logger, DispatchThread
os.environ['MKL_THREADING_LAYER'] = 'gnu'

gpus = list(range(8))
# gpus = [5, 6, 7]

lr = 2e-6
steps = 250
num_layers = 14
dataset = 'alpaca'

seed_list = [42, 43, 44, 45]
seed_list = [42]

sortby_list = ['alpha', 'layer', 'random', 'full']
sortby_list = ['alpha']

dataset_list = ['alpaca', 'oasst1', 'self-instruct']
orders = ['True', 'False']

ascending_order = 'True'
grid = itertools.product(seed_list, sortby_list)

# model = 'meta-llama/Meta-Llama-3-8B'
model = 'meta-llama/Llama-2-7b-hf'
cachedir = "/scratch/kinshuk/cache"
logger = get_logger('log', 'schedule_subspace.log')

# Bash command list
BASH_COMMAND_LIST = []

for seed, sortby in grid:
    
    save_path = "/jumbo/yaoqingyang/kinshuk/LlaMAft/results"

    cmd = (
        "OMP_NUM_THREADS=1 python /jumbo/yaoqingyang/kinshuk/LlaMAft/llamaft.py"
        f" --seed {seed}"
        f" --model_name_or_path {model}"
        f" --cache_dir {cachedir}"
        f" --output_dir {save_path}"
        f" --dataset {dataset}"
        f" --sortby {sortby}"
        f" --sort_ascending {ascending_order}"
        f" --num_layers {num_layers}"
        f" --max_steps {steps}"
        f" --data_seed 42"
        f" --source_max_len 1024"
        f" --learning_rate 2e-6"
        f" --dataloader_num_workers 1"
        f" --evaluation_strategy steps"
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