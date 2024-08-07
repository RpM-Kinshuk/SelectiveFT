import os
import itertools
from gputracker.gputracker import get_logger, DispatchThread
os.environ['MKL_THREADING_LAYER'] = 'gnu'

gpus = list(range(8))
# gpus = [5, 6, 7]

lr = 2e-6
steps = 200000
num_layers = 14

seed = 42
seed_list = [42, 43, 44, 45]

sortby_list = ['alpha_mid', 'layer', 'random']

dataset = 'glue'
dataset_list = ['alpaca', 'oasst1', 'self-instruct']

task_name = 'cola'
task_list = ['mrpc', 'cola', 'rte']

order = 'False'
order_list = ['True', 'False']

grid = itertools.product(task_list, sortby_list, order_list)
# model = 'meta-llama/Meta-Llama-3-8B'
model = 'meta-llama/Llama-2-7b-hf'
cachedir = "/scratch/kinshuk/cache"
logger = get_logger('log', 'schedule_subspace.log')

# Bash command list
BASH_COMMAND_LIST = []

for task_name, sortby, order in grid:
    
    save_path = "/jumbo/yaoqingyang/kinshuk/LlaMAft/output"

    cmd = (
        "OMP_NUM_THREADS=1 python /jumbo/yaoqingyang/kinshuk/LlaMAft/llamaft.py"
        f" --seed {seed}"
        f" --model_name_or_path {model}"
        f" --cache_dir {cachedir}"
        f" --output_dir {save_path}"
        f" --dataset {dataset}"
        f" --task_name {task_name}"
        f" --sortby {sortby}"
        f" --sort_ascending {order}"
        f" --num_layers {num_layers}"
        f" --max_steps {steps}"
        f" --data_seed 42"
        f" --source_max_len 1024"
        f" --learning_rate {lr}"
        f" --dataloader_num_workers 1"
        f" --evaluation_strategy steps"
    )

    BASH_COMMAND_LIST.append(cmd)

# Dispatch thread setup
dispatch_thread = DispatchThread(
    "Llama training",
    BASH_COMMAND_LIST,
    logger,
    gpu_m_th=500,
    gpu_list=gpus,
    maxcheck=0,
    num_gpus_needed=4
)

# Start and join the dispatch thread
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")