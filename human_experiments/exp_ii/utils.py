"""Utility functions which are not clearly part of other classes."""

import copy
import random
import time
import numpy as np

from psychopy import gui, monitors, visual, event
import os
import pandas as pd
import config as cfg
from typing import Tuple


def request_subject_info() -> Tuple[int, str]:
    """User Interface to enter subject information (subject number, age, gender,
    handedness and visual aid)

    Returns:
        subject_id (int): positive integer to identify the subject
        subject_data_path (str): file path where to save the subject's data
    """
    # create dummy app to make sure wx doesn't crash
    dialog = gui.Dlg(title="Subject Information")
    dialog.addField("Subject number:")
    dialog.addField("Age:")
    dialog.addField(
        "Gender:", choices=["female", "male", "diverse", "prefer to not disclose"]
    )
    dialog.addField("Handedness:", choices=["right", "left"])
    dialog.addField("Visual aid:", choices=["none", "glasses", "contact lenses"])

    valid = False

    # show dialogue box until a valid user response was given
    while not valid:
        dialogue_data = dialog.show()  # show dialog and wait for OK or Cancel

        if dialog.OK:
            print(dialogue_data)

            # test whether age is a positive integer > 0
            if (dialogue_data[1].isdigit()) and (int(dialogue_data[1]) > 0):
                age = int(dialogue_data[1])
            else:
                dialog.addText("Please enter a valid age (positive integer).")
                continue

            # test whether a positive integer was entered as subject number
            if dialogue_data[0].isdigit():
                subject_id = int(dialogue_data[0])

                # path for the subject's data (if needed add leading 0 for tidiness)
                subject_data_path = os.path.join(
                    cfg.data_folder, f"subject{subject_id:02}"
                )

                # test whether subject folder already exists (if yes, warn the user)
                if os.path.isdir(subject_data_path):
                    overwrite_existing_dialog = gui.Dlg("Warning")
                    overwrite_existing_dialog.addText(
                        f"Folder for subject {subject_id} already exists. "
                        f"Do you want to continue?"
                    )
                    overwrite_existing_dialog.show()

                    # if user still wants to continue, the subject data
                    # will be overwritten
                    if overwrite_existing_dialog.OK:
                        # remove entry with the subject_id which is going to be
                        # overwritten from the demographics file
                        demographics = pd.read_csv(cfg.demographics_file)
                        demographics = demographics[
                            demographics["subject_id"] != subject_id
                        ]
                        demographics.to_csv(cfg.demographics_file, index=False)
                        valid = True
                    else:
                        pass

                # if subject folder does not exist yet, create one
                else:
                    os.mkdir(subject_data_path)
                    print("Created subject directory.")
                    valid = True

            # if the input was no valid subject number, ask again for a valid number
            else:
                dialog.addText(
                    "Please enter a valid subject number (positive integer)."
                )

        else:
            raise Exception("User cancelled the experiment.")

    if valid:
        # save demographic information in a separate file
        subject_data = {
            "subject_id": subject_id,
            "age": age,
            "gender": dialogue_data[2],
            "handedness": dialogue_data[3],
            "visual_aid": dialogue_data[4],
            "date": time.strftime("%d.%m.%Y"),
        }
        df = pd.DataFrame(subject_data, index=[0])
        df.to_csv(cfg.demographics_file, index=False, mode="a", header=False)
        print("Saved demographic information!")

    return subject_id, subject_data_path


def setup_psychopy() -> Tuple[visual.Window, event.Mouse]:
    """Initializes the most fundamental components like the monitor,
    mouse and clock."""

    # setup monitor
    mon = monitors.Monitor("testMonitor")

    mon.setSizePix(cfg.screen_size)
    mon.setWidth(28.4)
    mon.setDistance(57)
    # window_size = mon.getSizePix()
    # print(mon.getSizePix(), mon.getWidth(), mon.getDistance())

    # create window object and mouse
    # window = visual.Window(size=config.screen_size, fullscr=True,
    # monitor=mon, units="pix")
    window = visual.Window(cfg.screen_size, monitor=mon, units="pix",
                           fullscr=cfg.fullscreen)

    # setup mouse
    mouse = event.Mouse(visible=False, win=window)

    return window, mouse


def create_repeated_trials(
    experiment_data: pd.DataFrame, subject_data: pd.DataFrame
) -> pd.DataFrame:

    n_repeat_trials_per_instr_type_per_extreme_confidence = cfg.n_repeat_trials_per_instr_type // 2

    # only consider trials from the main experiment (no practice trials and no catch trials)
    main_trials = experiment_data[(experiment_data["trial_type"] == "main") & (experiment_data["catch_trial"] == False)]
    main_subject_data = subject_data[(subject_data["trial_type"] == "main") & (subject_data["catch_trial"] == False)]
    
    new_block_nr = []
    
    new_index = []

    # find last block and trial nr of the main experiment
    last_block_nr = max(subject_data["block_nr"])
    last_trial_nr = max(subject_data["trial_nr"])

    # randomize the order of the instruction types for the repeated trials
    shuffled_instr_types = copy.deepcopy(cfg.instruction_types)
    random.shuffle(shuffled_instr_types)

    extreme_confident_indices_all_instr_types = []

    # sample trials for each instr_type
    for nr, instr_type in enumerate(shuffled_instr_types):
        # only consider trials of the respective instruction type and
        # the corresponding block numbers
        if instr_type in ("optimized", "natural"):
            relevant_trials = main_trials[main_trials["instr_type"] == instr_type]
            unique_block_nr = np.unique(relevant_trials["block_nr"])
        else:
            # find all trials from the none instruction type
            relevant_trials = main_trials[
                (main_trials["instr_type"] == "none_pre")
                | (main_trials["instr_type"] == "none_post")
            ]
            unique_block_nr_all = np.unique(relevant_trials["block_nr"])
            # sample at most 3 blocks
            np.random.shuffle(unique_block_nr_all)
            unique_block_nr = unique_block_nr_all[:3]

        new_block_nr = new_block_nr + (
            [last_block_nr + 1 + nr] * cfg.n_repeat_trials_per_instr_type
        )
        
        new_index += np.arange(cfg.n_repeat_trials_per_instr_type).tolist()

        # order df according to increasing absolute confidence level
        relevant_trials_sorted = relevant_trials.iloc[relevant_trials["conf_rating"].abs().argsort()]

        # shuffle rows within confidence level
        # create np array of abolsute confidence ratings
        unique_abs_conf_ratings = np.unique(relevant_trials_sorted["conf_rating"].abs())
        idx_list_all_conf_ratings_sorted_shuffled_within = []
        # loop through conf ratings: get indices, shuffle them and append to list for all conf ratings
        for unique_abs_conf_rating_i in unique_abs_conf_ratings:
            idx_list_specific_to_conf_rating = []
            for this_idx in relevant_trials_sorted[relevant_trials_sorted["conf_rating"].abs() == unique_abs_conf_rating_i].index:
                idx_list_specific_to_conf_rating.append(this_idx)
            random.shuffle(idx_list_specific_to_conf_rating)
            idx_list_all_conf_ratings_sorted_shuffled_within.append(idx_list_specific_to_conf_rating)
        final_idx_list = [cur_idx for cur_list in idx_list_all_conf_ratings_sorted_shuffled_within for cur_idx in cur_list]

        # take first and last n trials, create one list and shuffle it. Then append to list over instr_types.
        least_confident_indices = final_idx_list[:n_repeat_trials_per_instr_type_per_extreme_confidence]
        most_confident_indices = final_idx_list[-n_repeat_trials_per_instr_type_per_extreme_confidence:]
        extreme_confident_indices_this_instr_types = least_confident_indices + most_confident_indices
        random.shuffle(extreme_confident_indices_this_instr_types)
        extreme_confident_indices_all_instr_types = extreme_confident_indices_all_instr_types + extreme_confident_indices_this_instr_types

    # create trial structure for the repeated trials using the list of indices from above
    columns = main_subject_data.columns
    repeated_trials = pd.DataFrame(columns=columns)
    
    if cfg.n_repeat_trials_per_instr_type == 0:
        return repeated_trials
    
    for repeated_trial_index_i in extreme_confident_indices_all_instr_types:
        current_row = copy.deepcopy(main_subject_data[main_subject_data.index == repeated_trial_index_i])
        repeated_trials = repeated_trials.append(current_row)

    repeated_trials = repeated_trials.rename(
        columns={"trial_nr": "old_trial_nr", "block_nr": "old_block_nr"}
    )
    repeated_trials["trial_type"] = "repeated"
    repeated_trials["block_nr"] = new_block_nr
    repeated_trials["batch_block"] += 4
    repeated_trials["trial_nr"] = range(
        last_trial_nr + 1, last_trial_nr + 1 + len(repeated_trials)
    )
    repeated_trials["index"] = new_index

    return repeated_trials
