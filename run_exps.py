import subprocess
import random

wandb_group = "hparam-tuning-v3"

excluded_machines = 'jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard18'
# excluded_machines = 'jagupard14,jagupard18'
launch_command = f"nlprun -x {excluded_machines} -p low -a rohan-base -c 5 -o job-out/"
base_command =  f"cd ~/imSitu && python modified_crf.py --wandb-group {wandb_group}"

static_options = "--training_epochs 200"
train_set_sizes = [20000, 35000, 50000, 75000, 100000]

completions = []
for _ in range(1):
    for size in train_set_sizes:
        str = f'--training_set_size {size}'
        completions += [str, str + ' --collapse_annotations majority']


print(f'Launching {len(completions)} runs:')
for i, completion in enumerate(completions):
    command = f"{launch_command}{random.randint(0, 10**10)} '{base_command} {static_options} {completion}'"
    print(command)
    # subprocess.run(command, shell=True)
