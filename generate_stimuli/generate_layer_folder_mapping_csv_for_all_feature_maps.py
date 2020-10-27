# Generate the csv file that contains the mapping of the layer, kernel_size and channel to
# the folder structure for all feature map.
# this is required by the natural_stimuli_save_all_activations script

import csv
import utils as ut

# parameters
pre_post_relu = "pre_relu"


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


# ## Write to csv
csv_filename = "stimuli_analysis_all_feature_maps_one_feature_map_per_file/layer_folder_mapping_sampled_trials.csv"
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

channel_number = -1
kernel_size_previous = "1x1"
for layer_number, layer_name in enumerate(ut.layer_names_experiment_i):
    for kernel_size_number, kernel_size in enumerate(ut.kernel_size_list):
        if kernel_size == "1x1":
            start = 0
        else:
            start = ut.idx_of_kernel_sizes_in_each_layer_dict[
                layer_name + "_" + kernel_size_previous
            ]
        stop = ut.idx_of_kernel_sizes_in_each_layer_dict[layer_name + "_" + kernel_size]
        for feature_map_number in range(start, stop):
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
        channel_number += 1