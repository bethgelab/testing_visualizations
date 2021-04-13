"""Configuration for Experiment I"""

import os
from collections import namedtuple


# instruction conditions
instruction_types = ["none", "optimized", "natural"]
n_instruction_types = len(instruction_types)

instruction_main_order_odd = ["none_pre", "natural", "optimized", "none_post"]
instruction_main_order_even = ["none_pre", "optimized", "natural", "none_post"]

# number of levels per layer, kernel size and channel (unit)
n_unit_levels = {"n_layers": 9, "n_kernels": 4, "n_channels": 2}

# warm-up trails
n_warm_up_trials = 3

layers = [f"layer_{i}" for i in range(9)]
conditions = ["none", "optimized", "natural"]
kernel_sizes = ["0", "1", "2", "3"]

n_catch_trials_per_block = 3
n_practice_trials_per_condition = 5
n_main_trials_per_block_in_natural_condition = 15
n_main_trials_per_block_in_optimized_condition = n_main_trials_per_block_in_natural_condition
n_main_trials_per_block_in_none_condition = 11
n_repeat_trials_per_instr_type = 6

TrialConfiguration = namedtuple("TrialConfiguration", ["trial_name", "trial_type", "length"])

structure = [
    # none type
    [
        TrialConfiguration("none_pre", "practice", n_practice_trials_per_condition),
        TrialConfiguration("none_pre", "main", n_main_trials_per_block_in_none_condition),
        TrialConfiguration("none_pre", "main", n_main_trials_per_block_in_none_condition),
    ],
    # natural/optimized type
    [
        TrialConfiguration("optimized", "practice", n_practice_trials_per_condition),
        TrialConfiguration("natural", "practice", n_practice_trials_per_condition),

        TrialConfiguration("optimized", "main", n_main_trials_per_block_in_optimized_condition),
        TrialConfiguration("natural", "main", n_main_trials_per_block_in_natural_condition),
        TrialConfiguration("optimized", "main", n_main_trials_per_block_in_optimized_condition),
        TrialConfiguration("natural", "main", n_main_trials_per_block_in_natural_condition),
        TrialConfiguration("optimized", "main", n_main_trials_per_block_in_optimized_condition),
        TrialConfiguration("natural", "main", n_main_trials_per_block_in_natural_condition),
    ],
    # none type
    [
        TrialConfiguration("none_post", "main", n_main_trials_per_block_in_none_condition),
        TrialConfiguration("none_post", "main", n_main_trials_per_block_in_none_condition),
    ],
    # repeat type
    [
        TrialConfiguration("none_*", "repeat", n_repeat_trials_per_instr_type),
        TrialConfiguration("optimized", "repeat", n_repeat_trials_per_instr_type),
        TrialConfiguration("natural", "repeat", n_repeat_trials_per_instr_type),
    ]
]

# order of the instruction types for the practice trials (counterbalanced over subjects)
instruction_practice_order_odd = ["none", "natural", "optimized"]
instruction_practice_order_even = ["none", "optimized", "natural"]

# set the number of test and instruction patches
n_test_patches = 2
n_instruction_patches = 9  # for min and max each

# set the number of confidence ratings
n_conf_ratings = 6

fullscreen = True
screen_size = [1920, 1080]  # Full HD

seed = 42

data_folder = "../data"
demographics_file = os.path.join(data_folder, "demographics.csv")
stimuli_folder = "../stimuli_August_2020_20_batches/channel"
