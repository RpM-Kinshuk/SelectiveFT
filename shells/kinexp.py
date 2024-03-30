import os
os.environ['export MKL_THREADING_LAYER']='gnu'
import itertools

from numpy import sort
from sklearn.neighbors import sort_graph_by_row_values
from gputracker.gputracker import get_logger, DispatchThread

gpus = list(range(8))
gpus  = [5, 6, 7]
train_layers = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 30, 36, 72, 74]
task_list = ['cola', 'mrpc', 'rte', 'stsb', 'sst2', 'qnli', 'mnli', 'qqp']

model = "bert-base-uncased"
ascending_order = ["True", "False"]
sortby = ["alpha", "layer"]

norm = "False"
freeze_bert = "True"
train_seed_lst = [42, 357]
seed = 7
max_length = 128
batch_size = 32
epochs = 3
logger = get_logger('log', 'schedule_subspace.log')

grid = list(
    itertools.product(
        # train_seed_lst,
        task_list,
        # sortby,
        # ascending_order,
        train_layers,
    )
)
BASH_COMMAND_LIST = []
task = "sst2"
sby = "alpha"
order  = "False"
layers = 10
kqv = "True"
steps = 2

for layers in train_layers:
    
    save_path = "/rscratch/tpang/kinshuk/RpMKin/llama_ft/data"

    cmd = "OMP_NUM_THREADS=1 python /rscratch/tpang/kinshuk/RpMKin/llama_ft/LlaMAft/llamaft.py " + \
        f"--seed {seed} " + \
        f"--data_seed {seed} " + \
        f"--output_dir {save_path} " + \
        f"--dataset alpaca " + \
        f"--max_eval_samples 50 " + \
        f"--dataloader_num_workers 1 " + \
        f"--do_eval false " + \
        f"--max_steps {steps} " + \
        f"--sortby {sby} " + \
        f"--num_layers {layers} " + \
        f"--source_max_len 512 " + \
        f"--verbose " + \
        f"--memlog " + \
        f"--per_device_train_batch_size {batch_size} "
        
    BASH_COMMAND_LIST.append(cmd)


dispatch_thread = DispatchThread(
    "Alpaca dataset training",
    BASH_COMMAND_LIST,
    logger,
    gpu_m_th=5000,
    gpu_list=gpus,
    maxcheck=0,
)

dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")