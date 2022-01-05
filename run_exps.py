import subprocess
import random

wandb_group = "hparam-tuning-v2"

excluded_machines = 'jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard18'
# excluded_machines = 'jagupard14,jagupard18'
launch_command = f"nlprun -x {excluded_machines} -p standard -a rohan-base -c 5 -o job-out/"
base_command =  f"cd ~/imSitu && python modified_crf.py --wandb-group {wandb_group}"

static_options = "--training_epochs 200 --cnn_type resnet_18"
# train_set_sizes = [20000, 40000, 75000]

completions = [
"--merge_only_train_set --fix_dev_set_as_test_set --training_set_size 75000", # control
"--merge_only_train_set --fix_dev_set_as_test_set --training_set_size 40000", # subsample train
"--merge_only_train_set --training_set_size 40000 --test_set_size 25000", # shuffle split train
"--merge_only_train_and_dev_set --training_set_size 40000 --test_set_size 25000", # shuffle split train + dev
"--training_set_size 40000 --test_set_size 25000", # shuffle split train + dev + test
"--training_set_size 75000 --test_set_size 25000", # shuffle split train + dev + test with larger train set

# class balanced
"--merge_only_train_set --training_set_size 40000 --class_balance_test_set --test_set_imgs_per_class 50", # shuffle split train
"--merge_only_train_and_dev_set --training_set_size 40000 --class_balance_test_set --test_set_imgs_per_class 50", # shuffle split train + dev
"--training_set_size 40000 --class_balance_test_set --test_set_imgs_per_class 50", # shuffle split train + dev + test
"--training_set_size 75000 --class_balance_test_set --test_set_imgs_per_class 50", # shuffle split train + dev + test with larger train set
]

# for _ in range(1):
#     for size in train_set_sizes:
#         str = f'--training_set_size {size}'
#         completions += [str, str + ' --collapse_annotations majority']

print(f'Launching {len(completions)} runs:')
for i, completion in enumerate(completions):
    command = f"{launch_command}{random.randint(0, 10**10)} '{base_command} {static_options} {completion}'"
    print(command)
    # subprocess.run(command, shell=True)
