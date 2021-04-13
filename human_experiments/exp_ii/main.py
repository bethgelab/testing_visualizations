"""Main file for Experiment II."""

import os
import random
import pandas as pd
import numpy as np
import shutil
import config as cfg  # import custom config file
import utils as ut
import experiment_utils
from experiment import Experiment


def prepare_output():
    # Test if there is already a directory for the experimental data,
    # otherwise create one
    if not os.path.isdir(cfg.data_folder):
        os.mkdir(cfg.data_folder)
        print("Created data directory.")

    # Check whether there is already a csv file for the demographic
    # information, otherwise create one
    if not os.path.isfile(cfg.demographics_file):
        header = pd.DataFrame(
            columns=["subject_id", "age", "gender", "handedness", "visual_aid", "date"]
        )
        header.to_csv(cfg.demographics_file, index=False)
        print("Created demographics file.")


def run_experiment():
    # Code to run the experiment

    # enter subject id and save subject demographics in separate file
    subject_id, data_path = ut.request_subject_info()

    shutil.copy(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"),
        os.path.join(data_path, "config.py")
    )

    # We ran the experiment with the following seeds.
    # If you want to randomize across subjects, comment the seeds below out.
    random.seed(cfg.seed + subject_id)
    np.random.seed(cfg.seed + subject_id)
    # the seeds below are helpful for debugging
    # random.seed(cfg.seed)
    # np.random.seed(cfg.seed)

    # create subject trial structure
    subject_data = experiment_utils.build_experiment_structure(subject_id)
    subject_data.to_csv(
        os.path.join(data_path, "experiment_structure.csv"), index=False
    )

    # create empty DataFrame for experimental data
    initial_df = pd.DataFrame(
        columns=[
            # data from experiment_structure.csv
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
            "channel",
            # data from trial
            "query_order",
            "correct",
            "conf_rating",
            "RT",
            "timeout",
            "response",
            "correct_response",
            "query_image_a",
            "query_image_b",
            # in case of repeated trials
            "old_trial_nr",
            "old_block_nr",
            "seed",
        ]
    )
    initial_df.to_csv(os.path.join(data_path, "experiment_data.csv"), index=False)

    experiment = Experiment(subject_data, data_path)
    experiment.execute()


def main():
    prepare_output()

    run_experiment()


if __name__ == "__main__":
    main()
