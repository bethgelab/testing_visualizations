"""Utility functions used to create & manage the experimental data structure."""

import config as cfg
import pandas as pd
import numpy as np
from typing import Tuple, List
import random
import re
import copy


def _get_units_per_layer() -> Tuple[str, str, str]:
    def _get_units_per_kernel(kernel_size: str) -> Tuple[str, str]:
        if kernel_size == "3":
            return "channel_0", "channel_1"
        else:
            return "channel_0",

    return [(kernel_size, it) for kernel_size in cfg.kernel_sizes for it in
            _get_units_per_kernel(kernel_size)]


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
    # get all combinations for layer-kernel_size-feature_map and shuffle them
    layers_kernel_sizes_and_units = [(layer, *it) for layer in cfg.layers for it in
                                     _get_units_per_layer()]

    copy_none = layers_kernel_sizes_and_units.copy()
    copy_natural = layers_kernel_sizes_and_units.copy()
    copy_optimized = layers_kernel_sizes_and_units.copy()
    for copy_i in [copy_none, copy_natural, copy_optimized]:
        np.random.shuffle(copy_i)

    current_block = 0

    # add 5 entries for practice trials
    # use practice units
    practice_list_one = [
        ("layer_i", "i", "channel_i"), 
        ("layer_j", "j", "channel_j"),
        ("layer_k", "k", "channel_k"),
        ("layer_l", "l", "channel_l"),
        ("layer_m", "m", "channel_m")]
    practice_list_two = [
        ("layer_n", "n", "channel_n"),
        ("layer_o", "o", "channel_o"),
        ("layer_p", "p", "channel_p"),
        ("layer_q", "q", "channel_q"),
        ("layer_r", "r", "channel_r")]
    practice_list_three = [
        ("layer_s", "s", "channel_s"),
        ("layer_t", "t", "channel_t"),
        ("layer_u", "u", "channel_u"),
        ("layer_v", "v", "channel_v"),
        ("layer_w", "w", "channel_w")]
    random.shuffle(practice_list_one)
    random.shuffle(practice_list_two)
    random.shuffle(practice_list_three)
    random_lksus = [practice_list_one, practice_list_two, practice_list_three]
    
    random.shuffle(random_lksus)
    
    # for each condition (none, natural, optimized) concatenate practice and main trials 
    layers_kernel_sizes_and_units_pool = {
        "none_.*": random_lksus[0] + copy_none,
        "natural": random_lksus[1] + copy_natural,
        "optimized": random_lksus[2] + copy_optimized
    }

    subject_structure = copy.deepcopy(cfg.structure)

    # switch order of optimized and natural conditions for half of the subjects
    if subject_id % 2 == 0:
        # roll items per list
        subject_structure[1] = [x for t in zip(subject_structure[1][1::2],
                                               subject_structure[1][::2]) for x in t]

    all_trials = []
    # loop through none_pre, natural/optimized, none_post and trial
    for lst in subject_structure:
        # loop through practice and main blocks
        for it in lst:
            if it.trial_type == "repeat":
                continue

            # get relevant units from from big dict layers_kernel_sizes_and_units_pool
            lksu_key = _regex_key_match(layers_kernel_sizes_and_units_pool,
                                       it.trial_name, True)
            lksu = layers_kernel_sizes_and_units_pool[lksu_key]
            lksu_items = lksu[:it.length]
            assert len(lksu_items) == it.length
            # reduce the units in layers_kernel_sizes_and units_pool by the ones that were just used
            layers_kernel_sizes_and_units_pool[lksu_key] = lksu[it.length:]

            layers = [it[0] for it in lksu_items]
            kernel_sizes = [it[1] for it in lksu_items]
            channels = [it[2] for it in lksu_items]

            # for practice trials, use batch 0 for all participants
            if it.trial_type == "practice":
                current_batches = 0
            # for main trials, use a different batch for each condition and for each participant
            else:
                if 'none' in lksu_key:
                    current_batches = subject_id
                elif 'natural' in lksu_key:
                    current_batches = subject_id + 1
                    if current_batches == 20:
                        current_batches = 1
                elif 'optimized' in lksu_key:
                    current_batches = subject_id + 2
                    if current_batches == 20:
                        current_batches = 1
                    elif current_batches == 21:
                        current_batches = 2

            trials = pd.DataFrame(dict(
                subject_id=subject_id,
                instr_type=it.trial_name,
                catch_trial=False,
                batch=current_batches,
                trial_type=it.trial_type,
                layer=layers,
                kernel_size=kernel_sizes,
                channel=channels
            ))

            # add catch trials for optimized/natural
            if not it.trial_name.startswith("none") and it.trial_type == "main":
                catch_trials = trials.iloc[
                    np.random.choice(len(trials), cfg.n_catch_trials_per_block, replace=False)].copy()
                # overwrite layer, kernel_size, and channel column with units from practice and explanation units:
                catch_trial_pool_list = ["i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w"]
                chosen_catch_trials_list = random.sample(catch_trial_pool_list, cfg.n_catch_trials_per_block)
                
                list_counter = 0
                for row_i, row in catch_trials.iterrows():
                    catch_trials.at[row_i, "layer"] = "layer_" + chosen_catch_trials_list[list_counter]
                    catch_trials.at[row_i, "kernel_size"] = chosen_catch_trials_list[list_counter]
                    catch_trials.at[row_i, "channel"] = "channel_" + chosen_catch_trials_list[list_counter]
                    list_counter += 1
                    
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
        "instr_type",
        "trial_type",
        "catch_trial",
        "batch",
        "layer",
        "kernel_size",
        "channel"
        ]
    all_trials = all_trials[new_column_order]

    return all_trials
