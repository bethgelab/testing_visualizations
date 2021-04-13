"""Utility functions used to create & manage the experimental data structure."""

import config as cfg
import pandas as pd
import numpy as np
from typing import Tuple, List
import random
import re
import copy

"""
about batch blocks:
numbers+1 correspond to abcdefgh
0, 1, 2, 3 are used for main trials
4  is used for catch trials
5, 6 are used for practice
7 is not used
"""


def _get_units_per_layer() -> Tuple[str, str, str]:
    def _get_units_per_kernel(kernel_size: str) -> Tuple[str, str]:
        # only use random units
        return "channel_0"

    return [(kernel_size, _get_units_per_kernel(kernel_size)) for kernel_size in cfg.kernel_sizes]


def _regex_key_match(dct: dict, key: str, reverse=False) -> str:
    keys = list(dct.keys())
    matches = []
    for k in keys:
        assert isinstance(k, str), "Keys must be strings"
        if reverse:
            m = re.search(k, key)
        else:
            m = re.search(key, k)

        if m is not None:
            matches.append(k)

    assert len(matches) < 2, "Search as not unique and retrieved multiple keys"
    if len(matches) == 0:
        raise KeyError(key)

    return matches[0]


def _count_batches(structure: List[List[cfg.TrialConfiguration]],
                   n_catch_trials_per_block: int) -> int:
    count = 0
    for lst in structure:
        for it in lst:
            if it.trial_type == "repeat":
                continue

            count += it.length
            if it.trial_type == "main":
                if it.trial_name == "natural":
                    count += n_catch_trials_per_block

    return count


def build_experiment_structure(subject_id: int) -> pd.DataFrame:
    def get_lksu_pools():
        # get all combinations for layer-kernel_size-feature_map and shuffle them
        layers_kernel_sizes_and_units = [(layer, *it) for layer in cfg.layers for it in
                                         _get_units_per_layer()]

        copy_natural = layers_kernel_sizes_and_units.copy()
        copy_optimized = layers_kernel_sizes_and_units.copy()
        copy_natural_practice = layers_kernel_sizes_and_units.copy()
        copy_optimized_practice = layers_kernel_sizes_and_units.copy()
        for copy_i in [copy_natural, copy_optimized, copy_natural_practice, copy_optimized_practice]:
            np.random.shuffle(copy_i)

        layers_kernel_sizes_and_units_pool = {
            "natural": copy_natural,
            "optimized": copy_optimized
        }

        practice_layers_kernel_sizes_and_units_pool = {
            "natural": copy_natural_practice,
            "optimized": copy_optimized_practice,
        }

        return layers_kernel_sizes_and_units_pool, \
               practice_layers_kernel_sizes_and_units_pool

    original_subject_structure = copy.deepcopy(cfg.structure)

    # switch order of optimized and natural conditions for half of the subjects
    if ( (subject_id-1 ) // 4 ) % 2 == 0:
        # roll items per list
        original_subject_structure[0] = [x for t in zip(original_subject_structure[0][1::2],
                                               original_subject_structure[0][::2]) for x in t]
        original_subject_structure[1] = [x for t in zip(original_subject_structure[1][1::2],
                                               original_subject_structure[1][::2]) for x in t]
        original_subject_structure[2] = [x for t in zip(original_subject_structure[2][1::2],
                                               original_subject_structure[2][::2]) for x in t]
        original_subject_structure[3] = [x for t in zip(original_subject_structure[3][1::2],
                                               original_subject_structure[3][::2]) for x in t]

    # assign batch_block according to subject_id
    # tuples are in order of ablation conditions
    block_order = [
        (0, 1, 2, 3), # subject 0, 4, 8, ...
        (1, 2, 3, 0), # subject 1, 5, 9, ...
        (2, 3, 0, 1), # subject 2, 6, 10, ...
        (3, 0, 1, 2) # subject 3, 7, 11, ...
    ][(subject_id-1) % 4]
    subject_structure_with_batch_block = []
    for i, lst in enumerate(original_subject_structure):
        for j, it in enumerate(lst):
            if it.trial_type == 'main':
                subject_structure_with_batch_block.append(cfg.TrialConfiguration(*it[:4], block_order[i]))
            else:
                subject_structure_with_batch_block.append(cfg.TrialConfiguration(*it))

    # switch order of ablation conditions such that we have a uniform distribution over
    # the relevant orders and pick such that despite the numbering in the Kodierliste
    # the selection starts at the first entry
    if (subject_id-1) % 4 == 0:
        subject_structure = [
            subject_structure_with_batch_block[:6], # 0 max_max_1
            subject_structure_with_batch_block[6:12], # 1 min_max_1
            subject_structure_with_batch_block[12:18], # 2 max_max_9
            subject_structure_with_batch_block[18:] # 3 min_max_9
        ]
    elif (subject_id-1) % 4 == 1:
        subject_structure = [
            subject_structure_with_batch_block[:6], # 0 max_max_1
            subject_structure_with_batch_block[12:18], # 2 max_max_9
            subject_structure_with_batch_block[6:12], # 1 min_max_1
            subject_structure_with_batch_block[18:] # 3 min_max_9
        ]
    elif (subject_id-1) % 4 == 2:
        subject_structure = [
            subject_structure_with_batch_block[18:], # 3 min_max_9
            subject_structure_with_batch_block[6:12], # 1  min_max_1
            subject_structure_with_batch_block[12:18], # 2 max_max_9
            subject_structure_with_batch_block[:6] # 0 max_max_1
        ]
    elif (subject_id-1) % 4 == 3:
        subject_structure = [
            subject_structure_with_batch_block[18:], # 3 min_max_9
            subject_structure_with_batch_block[12:18], # 2 max_max_9
            subject_structure_with_batch_block[6:12], # 1 min_max_1
            subject_structure_with_batch_block[:6] # 0 max_max_1
        ]

    current_block = 0

    all_trials = []
    # loop through different condition types
    for lst in subject_structure:
        layers_kernel_sizes_and_units_pool, \
        practice_layers_kernel_sizes_and_units_pool = get_lksu_pools()

        # loop through practice and main blocks
        for it in lst:
            if it.trial_type == "repeat":
                continue

            lksu_dict = practice_layers_kernel_sizes_and_units_pool \
                if it.trial_type == "practice" else layers_kernel_sizes_and_units_pool

            # get relevant units from big dict layers_kernel_sizes_and_units_pool
            lksu_key = _regex_key_match(lksu_dict,
                                       it.trial_name, True)
            lksu = lksu_dict[lksu_key]
            lksu_items = lksu[:it.length]
            assert len(lksu_items) == it.length
            # reduce the units in layers_kernel_sizes_and units_pool by the ones that were just used
            lksu_dict[lksu_key] = lksu[it.length:]

            layers = [it[0] for it in lksu_items]
            kernel_sizes = [it[1] for it in lksu_items]
            channels = [it[2] for it in lksu_items]

            # for practice trials, use batch 0 for all participants
            if it.trial_type == "practice":
                current_batches = 0

            # for main trials, use a different batch for each condition and for
            # every fourth participant
            else:
                if 'natural' in lksu_key:
                    current_batches = ((subject_id - 1) // 4) * 2 + 1
                elif 'optimized' in lksu_key:
                    current_batches = ((subject_id - 1) // 4) * 2 + 2

            instruction_condition = it.instruction_condition

            trials = pd.DataFrame(dict(
                subject_id=subject_id,
                instr_type=it.trial_name,
                catch_trial=False,
                batch=current_batches,
                batch_block=it.batch_block,
                trial_type=it.trial_type,
                layer=layers,
                kernel_size=kernel_sizes,
                channel=channels,
                instruction_grid_mode=instruction_condition.grid_mode,
                n_instruction_patches=instruction_condition.n_patches,
            ))

            # add catch trials for optimized/natural
            if not it.trial_name.startswith("none") and it.trial_type == "main":
                catch_trials = trials.iloc[
                    np.random.choice(len(trials), cfg.n_catch_trials_per_block, replace=False)].copy()
                # overwrite batch block and thus effectively change the unit

                for row_i, _ in catch_trials.iterrows():
                    catch_trials.at[row_i, "batch_block"] = 4
                    
                catch_trials["catch_trial"] = True
                # batch 0 is just used for practice and catch trials
                catch_trials["batch"] = 0
                trials = pd.concat((trials, catch_trials))
            trials = trials.sample(frac=1).reset_index(drop=True)

            trials["block_nr"] = current_block
            current_block += 1

            all_trials.append(trials)

    all_trials = pd.concat(all_trials)
    all_trials = all_trials.reset_index()

    all_trials["trial_nr"] = range(len(all_trials))

    # rearrange column order to coincide with the column order of experiment_data.csv
    new_column_order = [
        "index",
        "subject_id",
        "trial_nr",
        "block_nr",
        "batch_block",
        "instr_type",
        "trial_type",
        "instruction_grid_mode",
        "n_instruction_patches",
        "catch_trial",
        "batch",
        "layer",
        "kernel_size",
        "channel"
        ]
    all_trials = all_trials[new_column_order]

    return all_trials
