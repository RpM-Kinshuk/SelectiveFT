import os
import itertools
from gputracker.gputracker import get_logger, DispatchThread
os.environ['MKL_THREADING_LAYER'] = 'gnu'

gpus = list(range(8))
# gpus = [5, 6, 7]

lr = 2e-5
lr_list = [2e-4, 2e-5, 2e-6, 2e-7]

steps = 200000
num_layers = 14
layer_list = [0, 2, 4, 8, 12, 18, 24, 32, 42, 64]

seed = 42
seed_list = [40, 41, 42]

sortby = 'alpha_mid'
sortby_list = ['alpha_mid', 'layer', 'random']

dataset = 'glue'
dataset_list = ['alpaca', 'oasst1', 'self-instruct']

task_name = 'cola'
task_list = ['mrpc', 'cola', 'rte']

order = 'False'
order_list = ['True', 'False']

grid = itertools.product(sortby_list, layer_list)
# model = 'meta-llama/Meta-Llama-3-8B'
model = 'meta-llama/Llama-2-7b-hf'
cachedir = "/scratch/kinshuk/cache"
logger = get_logger('log', 'schedule_subspace.log')

# Bash command list
BASH_COMMAND_LIST = []

for sortby, num_layers in grid:
    
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
        f" --verbose False"
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
    num_gpus_needed=1,
)

# Start and join the dispatch thread
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")