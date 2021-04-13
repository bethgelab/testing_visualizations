"""Configuration for Experiment II"""

import os
from collections import namedtuple


# instruction conditions
instruction_types = ["optimized", "natural"]
n_instruction_types = len(instruction_types)

# number of levels per layer, kernel size and channel (unit)
n_unit_levels = {"n_layers": 9, "n_kernels": 4, "n_channels": 2}

# warm-up trails
n_warm_up_trials = 3

layers = [f"layer_{i}" for i in [0, 2, 4, 6, 8]]
conditions = ["optimized", "natural"]
kernel_sizes = ["0", "1", "2", "3"]

n_catch_trials_per_block = 2
n_practice_trials_per_condition = 3
n_main_trials_per_block_in_natural_condition = 10
n_main_trials_per_block_in_optimized_condition = n_main_trials_per_block_in_natural_condition

AblationCondition = namedtuple(
    "AblationCondition",
    [
        "grid_mode",  # allowed values are: min_max, min_min, max_max, max_min
        "n_patches"
    ]
)
TrialConfiguration = namedtuple(
    "TrialConfiguration",
    ["trial_name", "trial_type", "length", "instruction_condition", "batch_block"]
)

ablation_condition_min_max_9 = AblationCondition("min_max", 9)
ablation_condition_min_max_1 = AblationCondition("min_max", 1)
ablation_condition_max_max_9 = AblationCondition("max_max", 9)
ablation_condition_max_max_1 = AblationCondition("max_max", 1)

structure = [
    [
        TrialConfiguration("optimized", "practice", n_practice_trials_per_condition, condition, 5),
        TrialConfiguration("natural", "practice", n_practice_trials_per_condition, condition, 6),

        TrialConfiguration("optimized", "main", n_main_trials_per_block_in_optimized_condition, condition, None),
        TrialConfiguration("natural", "main", n_main_trials_per_block_in_natural_condition, condition, None),
        TrialConfiguration("optimized", "main", n_main_trials_per_block_in_optimized_condition, condition, None),
        TrialConfiguration("natural", "main", n_main_trials_per_block_in_natural_condition, condition, None),
    ]
    # repeat experiment for all instruction conditions
    for condition in
    (
        ablation_condition_max_max_1, #1
        ablation_condition_min_max_1, #2
        ablation_condition_max_max_9, #3
        ablation_condition_min_max_9  #4
    )
]

n_instruction_patches_warm_up = 9

# set the number of confidence ratings
n_conf_ratings = 6

fullscreen = True
screen_size = [1920, 1080]

seed = 42

data_folder = "../data"
demographics_file = os.path.join(data_folder, "demographics.csv")
stimuli_folder_10 = "../stimuli_ablation_study_September_10_stimuli/"
stimuli_folder_2 = "../stimuli_ablation_study_September_2_stimuli/"
