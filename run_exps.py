import subprocess
import random

wandb_group = "feedback-v1"

excluded_machines = 'jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15'
excluded_machines = 'jagupard28'
launch_command = f"nlprun -x {excluded_machines} -a rohan-base -c 8 -r 25G -d 3090 -p high -o job-out/"
base_command =  f"cd ~/imSitu && python feedback_crf.py --wandb-group {wandb_group}"

static_options = ""

completions = [
    "--init-train-set-size 20000 --new-label-samples 2500 --new-unlabel-samples 2500",
    "--init-train-set-size 20000 --new-label-samples 1000 --new-unlabel-samples 4000",
    "--init-train-set-size 50000 --new-label-samples 2500 --new-unlabel-samples 2500",
]


print(f'Launching {len(completions)} runs:')
for i, completion in enumerate(completions):
    command = f"{launch_command}{random.randint(0, 10**10)} '{base_command} {static_options} {completion}'"
    print(command)
    # subprocess.run(command, shell=True)
