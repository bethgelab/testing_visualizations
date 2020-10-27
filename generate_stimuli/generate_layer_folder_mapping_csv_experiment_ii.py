# Generate the csv file that contains the mapping of the layer, kernel_size and channel
# to the folder structure for experiment II

import random
import csv
import itertools
import pandas as pd
import utils as ut
import shutil

# parameters
pre_post_relu = "pre_relu"
random.seed(28)


def get_random_feature_map_number(layer_name, kernel_size_previous, kernel_size):
    """get a random feature_map_number. It is sampled from the range defined by 
    the two kernels. Also, it is ensured that it does not correspond to one of 
    the warm-up trials"""
    if kernel_size == "1x1":
        start = 0
    else:
        start = ut.idx_of_kernel_sizes_in_each_layer_dict[
            layer_name + "_" + kernel_size_previous
        ]
    stop = (
        ut.idx_of_kernel_sizes_in_each_layer_dict[layer_name + "_" + kernel_size] - 1
    )  # exclude last int
    feature_map_number = random.randint(start, stop)

    # join all OrderedDicts with interpretable units
    # (for experiment ii: only warm-up trials)
    if layer_name in ut.feature_maps_judged_most_interpretable_warm_up.keys():
        all_interpretable_dicts = [
            ut.feature_maps_judged_most_interpretable_warm_up[layer_name]
        ]
    else:
        all_interpretable_dicts = []
    interpretable_units_list = list(set(itertools.chain(*all_interpretable_dicts)))

    # read in all csv files
    warm_up_trials_df = pd.read_csv(
        "stimuli_study_ii/layer_folder_mapping_warm_up_trials.csv", header=1
    )
    sampled_trials_a_df = pd.read_csv(
        "stimuli_study_ii/batch_block_a/layer_folder_mapping_sampled_trials.csv",
        header=1,
    )
    sampled_trials_b_df = pd.read_csv(
        "stimuli_study_ii/batch_block_b/layer_folder_mapping_sampled_trials.csv",
        header=1,
    )
    sampled_trials_c_df = pd.read_csv(
        "stimuli_study_ii/batch_block_c/layer_folder_mapping_sampled_trials.csv",
        header=1,
    )
    sampled_trials_d_df = pd.read_csv(
        "stimuli_study_ii/batch_block_d/layer_folder_mapping_sampled_trials.csv",
        header=1,
    )
    sampled_trials_e_df = pd.read_csv(
        "stimuli_study_ii/batch_block_e/layer_folder_mapping_sampled_trials.csv",
        header=1,
    )
    sampled_trials_f_df = pd.read_csv(
        "stimuli_study_ii/batch_block_f/layer_folder_mapping_sampled_trials.csv",
        header=1,
    )
    sampled_trials_g_df = pd.read_csv(
        "stimuli_study_ii/batch_block_g/layer_folder_mapping_sampled_trials.csv",
        header=1,
    )
    sampled_trials_h_df = pd.read_csv(
        "stimuli_study_ii/batch_block_h/layer_folder_mapping_sampled_trials.csv",
        header=1,
    )
    all_previous_trials = [
        warm_up_trials_df,
        sampled_trials_a_df,
        sampled_trials_b_df,
        sampled_trials_c_df,
        sampled_trials_d_df,
        sampled_trials_e_df,
        sampled_trials_f_df,
        sampled_trials_g_df,
        sampled_trials_h_df,
    ]
    all_previous_trials_df = pd.concat(all_previous_trials)
    cur_df = all_previous_trials_df[all_previous_trials_df.layer_name == layer_name]

    # conditions
    contained_in_warm_up = feature_map_number in interpretable_units_list
    contained_in_previous_trials = False
    for index, row in cur_df.iterrows():
        if (
            row.layer_name == layer_name
            and row.feature_map_number == feature_map_number
        ):
            contained_in_previous_trials = True

    if contained_in_warm_up or contained_in_previous_trials:
        feature_map_number = get_random_feature_map_number(
            layer_name, kernel_size_previous, kernel_size
        )

    return feature_map_number


# In[ ]:


def write_to_csv(
    csv_filename,
    pre_post_relu,
    layer_number,
    layer_name,
    kernel_size_number,
    kernel_size,
    channel_number,
    feature_map_number,
):
    """write specification to csv"""

    # save each layer-kernel-channel to csv
    with open(csv_filename, "a") as csvFile:  # 'a' for append
        csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
        csv_writer.writerow(
            [
                layer_number,
                kernel_size_number,
                channel_number,
                layer_name,
                pre_post_relu,
                kernel_size,
                feature_map_number,
            ]
        )
    csvFile.close()


# Write Sampled trials AND Practice, Explanation and Catch Trials to CSV


def generate_csv_file(batch_block):
    csv_filename = f"stimuli_study_ii/batch_block_{batch_block}/layer_folder_mapping_sampled_trials.csv"
    # write down what columns mean
    with open(csv_filename, "w") as csvFile:
        csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
        csv_writer.writerow(
            ["encoding in folder structure - correspondence in InceptionV1"]
        )
        csv_writer.writerow(
            [
                "layer_number",
                "kernel_size_number",
                "channel_number",
                "layer_name",
                "pre_post_relu",
                "kernel_size",
                "feature_map_number",
            ]
        )
    csvFile.close()


def generate_filling_for_csv_file(batch_block):
    csv_filename = f"stimuli_study_ii/batch_block_{batch_block}/layer_folder_mapping_sampled_trials.csv"

    kernel_size_previous = "1x1"
    for layer_number, layer_name in enumerate(ut.layer_names_experiment_i):
        if layer_name in ut.layer_names_experiment_ii:
            for kernel_size_number, kernel_size in enumerate(ut.kernel_size_list):
                # pick one random unit
                channel_number = 0
                feature_map_number = get_random_feature_map_number(
                    layer_name, kernel_size_previous, kernel_size
                )
                write_to_csv(
                    csv_filename,
                    pre_post_relu,
                    layer_number,
                    layer_name,
                    kernel_size_number,
                    kernel_size,
                    channel_number,
                    feature_map_number,
                )

                kernel_size_previous = kernel_size


# batch_block_a will be taken from experiment i
batch_block_list_to_generate_trials_for = ["b", "c", "d", "e", "f", "g", "h"]
for batch_block in batch_block_list_to_generate_trials_for:
    generate_csv_file(batch_block)
# copy batch_block_a from experiment i
shutil.copy(
    "stimuli_experiment_i/layer_folder_mapping_warm_up_trials.csv",
    "stimuli_study_ii/batch_block_a/layer_folder_mapping_sampled_trials.csv",
)


for batch_block in batch_block_list_to_generate_trials_for:
    print(batch_block)
    generate_filling_for_csv_file(batch_block)
