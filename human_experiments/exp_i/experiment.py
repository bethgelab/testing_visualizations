"""Experiment class."""

import random

import string
from psychopy import core, event
import numpy as np
import pandas as pd
import os
from typing_extensions import Final

import config as cfg
import utils as ut

from ui.trials.forced_choice_trial import ForcedChoiceTrial
from ui.trials.warmup_trial import WarmUpTrial
from ui import trials


class Experiment:
    """ Manages the data for one experiment and executes the experiment, i.e.
    shows the experiment to the subject and processes their responses.
    """

    def __init__(self, subject_data: pd.DataFrame, data_path: str):
        self.subject_data = subject_data
        self.data_path = data_path
        self.control_trials_data = None

        self.window, self.mouse = ut.setup_psychopy()

    def execute(self):
        """Execute the experiment by showing all trials and collecting the responses."""

        # present start screen
        self.mouse.setVisible(True)
        self.window.flip()
        core.wait(0.1)

        self._execute_introduction()
        self._execute_warm_up_trials()
        self._execute_instructions_main()
        self._execute_main_trials()

        # read in experimental data of the main experiment
        experiment_data = pd.read_csv(
            os.path.join(self.data_path, "experiment_data.csv")
        )

        # create trial structure for the repeated trials
        self.control_trials_data = ut.create_repeated_trials(
            experiment_data, self.subject_data
        )
        self.control_trials_data.to_csv(
            os.path.join(self.data_path, "repeated_trials_experiment_structure.csv"),
            index=False,
        )

        # execute repeated trials
        self._execute_control_trials()

        trials.KeypressTrial(
            self.mouse,
            self.window,
            keys=["space"],
            text="This is the end\nof the first part\nof the experiment.\n\n\n(Press `space` to continue.)",
        ).run()

        trials.KeypressTrial(
            self.mouse,
            self.window,
            keys=["space"],
            text="The experimenter\nwill now follow up\nwith two questions.\n\n\n(Press `space` to continue.)",
        ).run()

        trials.KeypressTrial(
            self.mouse,
            self.window,
            keys=["space"],
            text="Thank you for now.\n\n\n(Press space to exit)",
        ).run()

    def _execute_introduction(self):
        """Execute the trials which are part of the introduction."""

        trials.BatchedTrial(
            self.mouse,
            self.window,
            [
                trials.KeypressTrial(
                    self.mouse,
                    self.window,
                    keys=["space"],
                    text="Welcome to the experiment.\nIt will start soon.\n\n\n(Press `space` to continue.)",
                ),
                trials.KeypressTrial(
                    self.mouse,
                    self.window,
                    keys=["space"],
                    text="At first, we will ask you\nhow intuitive you find\nthe "
                    "interpretability method.\n\n\n(Press `space` to continue.)",
                ),
            ],
        ).run()

        event.clearEvents()

    def _execute_warm_up_trials(self):
        """Execute the warm-up (name in paper: intuitiveness) trials of the experiment."""

        trials.KeypressTrial(
            self.mouse,
            self.window,
            keys=["space"],
            text=f"Please move the slider\nto the left (not intuitive)\nor to the right (intuitive)\naccording to "
                 f"your impression.\n\n\n(Press `space` to continue.)",
        ).run()
        
        # Warm-up trials
        rating_list = []
        rt_list = []
        trial_nr_list_warm_up = []
        trial_name_list = []
        warm_up_trials = list(string.ascii_lowercase)[: cfg.n_warm_up_trials]
        
        # randomize the order of warm_up_trials
        random.shuffle(warm_up_trials)
        
        
        for trial_nr, trial_name in enumerate(warm_up_trials):
            folder_name = os.path.join(
                cfg.stimuli_folder,
                "warm_up_trials",
                f"layer_{trial_name}",
                f"kernel_size_{trial_name}",
                f"channel_{trial_name}"
            )
            optimized_stimuli = [
                os.path.join(folder_name, f"optimized_images", f"max_{i}.png")
                for i in range(cfg.n_instruction_patches)
            ]
            natural_stimuli = [
                os.path.join(folder_name, f"natural_images", f"max_{i}.png")
                for i in range(cfg.n_instruction_patches)
            ]

            trial = WarmUpTrial(
                self.mouse,
                self.window,
                optimized_stimuli,
                natural_stimuli,
                "Optimized Images",
                "Natural Images",
            )
            rating, rt = trial.run()
            rating_list.append(int(rating))
            rt_list.append(rt)
            trial_nr_list_warm_up.append(trial_nr)
            trial_name_list.append(trial_name)

        warm_up_data_dict = {
            "subject_id": self.subject_data.subject_id[0],
            "trial_name": trial_name_list,
            "trial_nr": trial_nr_list_warm_up,
            "rating": rating_list,
            "rt": rt_list,
        }

        # save warm-up data in csv-file
        warm_up_data = pd.DataFrame(warm_up_data_dict)
        warm_up_data.to_csv(
            os.path.join(self.data_path, "warm_up_experiment_data.csv"), index=False
        )

    def _execute_instructions_main(self):
        """Execute the instructions for the main experiment."""

        trials.BatchedTrial(
            self.mouse,
            self.window,
            [
                trials.KeypressTrial(
                    self.mouse,
                    self.window,
                    keys=["space"],
                    text="From now on,\nyou will always see\nthe following layout of the screen.\n\n\n"
                         "(Press `space` to continue.)",
                ),
                trials.KeypressTrial(
                    self.mouse,
                    self.window,
                    keys=["space"],
                    text="On the sides of the screen,\nyou will either see\nno instruction images,\nnatural instruction"
                         " images or\noptimized instruction images.\n\nAt the center of the screen,\nyou will always "
                         "see\nnatural images.\n\n\n(Press `space` to continue)",
                ),
                trials.KeypressTrial(
                    self.mouse,
                    self.window,
                    keys=["space"],
                    text="The question is always\nthe following:\n\n\n(Press `space` to continue.)",
                ),
                trials.KeypressTrial(
                    self.mouse,
                    self.window,
                    keys=["space"],
                    text="Which of the two images\nat the center of the screen\nis also a\nmaximally activating image?"
                         "\n\n\n(Press `space` to continue.)",
                ),
                
            ],
        ).run()

    def _execute_main_trials(self):
        """Execute the main trials of the experiment."""

        self._execute_main_type_trials(self.subject_data)

    def _execute_main_type_trials(self, trials_data: pd.DataFrame):
        """Execute the given set of trials (which are either main or
        repeated main trials)."""

        initial_trial_nr = trials_data.iloc[0]["trial_nr"]

        previous_block = trials_data.iloc[0]["block_nr"]
        cum_sum_previous_block_trials = initial_trial_nr
        next_block = False

        correct_list = []

        def show_feedback(trial_nr):
            nonlocal cum_sum_previous_block_trials
            # display how many trials were correct in the last block
            n_last_block_trials = trial_nr - cum_sum_previous_block_trials
            n_last_block_correct_trials = sum(correct_list[-n_last_block_trials:])

            cum_sum_previous_block_trials = trial_nr

            trials.KeypressTrial(
                self.mouse,
                self.window,
                keys=["space"],
                text=f"You got {n_last_block_correct_trials} out of "
                     f"{n_last_block_trials} trials correct.\n\n\n(Press `space` to continue.)",
            ).run()

        for row in range(len(trials_data)):
            trial_info = trials_data.iloc[row]
            trial_nr = trial_info["trial_nr"]
            trial_type = trial_info["trial_type"]
            instr_type = trial_info["instr_type"]
            subject_id = trial_info["subject_id"]
            batch = trial_info["batch"]
            block = int(trial_info["block_nr"])
            catch_trial = trial_info["catch_trial"]

            trial_nr_in_block = trial_info["index"]
            n_trials_in_block = trials_data[trials_data["block_nr"] == block]["index"].max() + 1

            catch_trial = not np.isnan(catch_trial) and bool(catch_trial)

            if trial_type == "repeated":
                old_trial_nr = trial_info["old_trial_nr"]
                old_block_nr = trial_info["old_block_nr"]
            else:
                old_trial_nr = np.NaN
                old_block_nr = np.NaN

            # Check if a block was finished (which means it is time for
            # feedback about the last block and a break)
            if block != previous_block:
                show_feedback(trial_nr)

                while True:
                    keys = trials.KeypressTrial(
                        self.mouse,
                        self.window,
                        keys=["space", "escape"],
                        text="Short break.\n\n\n(Press `space` to continue.)",
                    ).run()

                    if len(keys) == 1 and "space" in keys:
                        break
                    elif len(keys) == 1 and "escape" in keys:
                        self.window.fullscr = not self.window.fullscr
                        self.window.flip()
                        print("Subject toggled fullscreen")

                next_block = True
                previous_block = block

            # If a new block was started, display info about instruction type
            # of the stimuli
            if next_block or trial_nr == initial_trial_nr:
                instruction_title = {
                    "none_pre": "No",
                    "none": "No",
                    "none_post": "No",
                    "optimized": "Optimized",
                    "natural": "Natural",
                }[instr_type]
                trials.KeypressTrial(
                    self.mouse,
                    self.window,
                    keys=["space"],
                    text=f"Now you will see the\nfollowing type of\ninstruction images:\n\n"
                         f"{instruction_title} images\n\n\n(Press `space` to start the trials.)",
                ).run()
                trials.KeypressTrial(
                    self.mouse,
                    self.window,
                    keys=["space"],
                    text="The question is still the same:\nWhich of the two images\nat the center of the screen\nis "
                         "also a\nmaximally activating image?\n\n\n(Press `space` to continue.)",
                ).run()

                next_block = False

            layer = trial_info["layer"]
            kernel_size = trial_info["kernel_size"]
            channel = trial_info["channel"]
            folder_name = os.path.join(
                cfg.stimuli_folder,
                "sampled_trials",
                f"{layer}",
                f"kernel_size_{kernel_size}",
                f"{channel}",
            )
            query_folder_name = os.path.join(folder_name, "natural_images",
                                             f"batch_{batch}")
            if instr_type in ("optimized", "natural"):
                if instr_type == "optimized":
                    instruction_folder_name = os.path.join(
                        folder_name, "optimized_images"
                    )
                elif instr_type == "natural":
                    instruction_folder_name = os.path.join(
                        folder_name, "natural_images", f"batch_{batch}"
                    )

                min_stimuli = [
                    os.path.join(instruction_folder_name, f"min_{i+1}.png")
                    for i in range(cfg.n_instruction_patches)
                ]
                max_stimuli = [
                    os.path.join(instruction_folder_name, f"max_{i}.png")
                    for i in range(cfg.n_instruction_patches)
                ]
                min_title = "Minimally activating"
                max_title = "Maximally activating"
            else:
                min_stimuli = []
                max_stimuli = []

                min_title = ""
                max_title = ""

            if catch_trial:
                query_image_a = np.random.choice(min_stimuli, 1).item()
                query_image_b = np.random.choice(max_stimuli, 1).item()
            else:
                query_image_a = os.path.join(query_folder_name, "min_0.png")
                query_image_b = os.path.join(query_folder_name, "max_9.png")

            # flip order of query images randomly
            if np.random.rand() < 0.5:
                query_order = "min_max"
                correct_response: Final = "b"
            else:
                query_order = "max_min"
                correct_response: Final = "a"
                query_image_a, query_image_b = query_image_b, query_image_a

            trial = ForcedChoiceTrial(
                self.mouse,
                self.window,
                min_stimuli,
                max_stimuli,
                min_title,
                max_title,
                query_image_a,
                query_image_b,
                # timeout=timeout_threshold,
                global_progress=(trial_nr_in_block + 1 ) / (n_trials_in_block + 1),
                correct=correct_response,
            )

            rt, response, confidence, timeout_reached = trial.run()
            correct = response == correct_response

            correct_list.append(correct)

            trial_data = {
                # data from experiment_structure.csv
                "subject_id": subject_id,
                "trial_nr": trial_nr,
                "block_nr": block,
                "instr_type": instr_type,
                "trial_type": trial_type,
                "catch_trial": catch_trial,
                "batch": batch,
                "layer": layer,
                "kernel_size": kernel_size,
                "channel": channel,
                # data from trial
                "query_order": query_order,
                "correct": correct,
                "conf_rating": confidence,
                "RT": rt,
                "timeout": timeout_reached,
                "response": response,
                "correct_response": correct_response,
                "query_image_a": query_image_a,
                "query_image_b": query_image_b,
                # in case of repeated trials
                "old_trial_nr": old_trial_nr,
                "old_block_nr": old_block_nr,
                "seed": cfg.seed,
            }

            trial_data_df = pd.DataFrame(trial_data, index=[0])
            trial_data_df.to_csv(
                os.path.join(self.data_path, "experiment_data.csv"),
                index=False,
                mode="a",
                header=False,
            )

        show_feedback(trial_nr + 1)

    def _execute_control_trials(self):
        """Execute the control trials of the experiment."""

        assert self.control_trials_data is not None

        self._execute_main_type_trials(self.control_trials_data)
