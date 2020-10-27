# Generate the csv file that contains the mapping of the layer, kernel_size and channel to the folder structure

import random
import csv
import itertools
import utils as ut

# parameters
pre_post_relu = "pre_relu"
random.seed(28)

sampled_trials_csv_filename = (
    "stimuli_experiment_i/layer_folder_mapping_sampled_trials.csv"
)
warmup_trials_csv_filename = (
    "stimuli_experiment_i/layer_folder_mapping_warm_up_trials.csv"
)
practice_csv_filename = (
    "stimuli_experiment_i/layer_folder_mapping_practice_and_explanation_trials.csv"
)


def get_random_feature_map_number(layer_name, kernel_size_previous, kernel_size):
    """get a random feature_map_number. It is sampled from the range defined by 
    the two kernels. Also, it is ensured that it does not correspond to one of the 
    feature_maps that were considered interpretable by Olah et al."""
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
    all_interpretable_dicts = [
        ut.feature_maps_judged_interpretable_by_Olah[layer_name],
        ut.feature_maps_judged_most_interpretable_sampled_trials[layer_name],
    ]

    interpretable_units_list = list(set(itertools.chain(*all_interpretable_dicts)))

    if feature_map_number in interpretable_units_list:
        feature_map_number = get_random_feature_map_number(
            layer_name, kernel_size_previous, kernel_size
        )

    return feature_map_number


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
    with open(csv_filename, "a") as csvFile:
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


def create_csv_file(csv_filename):
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


# for sampled trials
create_csv_file(sampled_trials_csv_filename)

kernel_size_previous = "1x1"
for layer_number, layer_name in enumerate(ut.layer_names_experiment_i):
    for kernel_size_number, kernel_size in enumerate(ut.kernel_size_list):
        # pick one random unit
        channel_number = 0
        feature_map_number = get_random_feature_map_number(
            layer_name, kernel_size_previous, kernel_size
        )
        write_to_csv(
            sampled_trials_csv_filename,
            pre_post_relu,
            layer_number,
            layer_name,
            kernel_size_number,
            kernel_size,
            channel_number,
            feature_map_number,
        )

        # add one interpretable unit
        if kernel_size == "pool":
            channel_number = 1
            feature_map_number = ut.feature_maps_judged_most_interpretable_sampled_trials[
                layer_name
            ][
                0
            ]
            write_to_csv(
                sampled_trials_csv_filename,
                pre_post_relu,
                layer_number,
                layer_name,
                kernel_size_number,
                kernel_size,
                channel_number,
                feature_map_number,
            )

        kernel_size_previous = kernel_size


# for warmup trials
create_csv_file(warmup_trials_csv_filename)

letter_list = ["a", "b", "c"]

for layer_number, (layer_name, feature_map_number) in enumerate(
    ut.feature_maps_judged_most_interpretable_warm_up.items()
):
    print(layer_number, layer_name)
    layer_letter = letter_list[layer_number]
    kernel_size_letter = letter_list[layer_number]
    channel_letter = letter_list[layer_number]
    kernel_size = ut.get_kernel_size(feature_map_number[0], layer_name)
    write_to_csv(
        warmup_trials_csv_filename,
        pre_post_relu,
        layer_letter,
        layer_name,
        kernel_size_letter,
        kernel_size,
        channel_letter,
        feature_map_number[0],
    )

# for practice trials
kernel_size_dict = {"1x1": 0, "3x3": 1, "5x5": 2, "pool": 3}
layer_names_list = ut.layer_names_experiment_i
kernel_size_list = ut.kernel_size_list


kernel_size_previous = "1x1"
for practice_evaluation_i in range(4):
    for layer_number, layer_name in enumerate(random.sample(layer_names_list, 5)):
        kernel_size = random.sample(kernel_size_list, 1)[0]
        if kernel_size == "1x1":
            kernel_size_previous = "1x1"
        elif kernel_size == "3x3":
            kernel_size_previous = "1x1"
        elif kernel_size == "5x5":
            kernel_size_previous = "3x3"
        elif kernel_size == "pool":
            kernel_size_previous = "5x5"
        kernel_size_number = kernel_size_dict[kernel_size]
        # pick one random unit
        channel_number = 2
        feature_map_number = get_random_feature_map_number(
            layer_name, kernel_size_previous, kernel_size
        )
        write_to_csv(
            practice_csv_filename,
            pre_post_relu,
            layer_number,
            layer_name,
            kernel_size_number,
            kernel_size,
            channel_number,
            feature_map_number,
        )
