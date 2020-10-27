import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import PIL


def make_all_dirs(
    stimuli_dir,
    objective,
    trial_type,
    image_type,
    unit_specs_df,
    n_batches_stop=3,
    batch_dirs=True,
):
    """If they do not exist yet, make all directories for the unit specifications
    and - in case of natural images - for all the batches"""

    for idx, row in unit_specs_df.iterrows():
        stim_dir_super = get_stim_dir_super(
            stimuli_dir, objective, trial_type, row, image_type
        )
        if not os.path.exists(stim_dir_super):
            os.makedirs(stim_dir_super)

        if image_type == "natural" and batch_dirs == True:
            make_dir_for_all_batches(stim_dir_super, n_batches_stop)


def get_stim_dir_super(stimuli_dir, objective, trial_type, row, image_type):

    stim_dir_super = os.path.join(
        stimuli_dir,
        objective,
        trial_type,
        f"layer_{row['layer_number']}",
        f"kernel_size_{row['kernel_size_number']}",
        f"channel_{row['channel_number']}",
        f"{image_type}_images",
    )

    return stim_dir_super


def remove_white_margins(ax):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())


###### SPECIFIC TO NATURAL STIMULI ######


def make_batch_dir(stimuli_dir, objective, trial_type, row, image_type, batch_number):
    stim_dir_super = get_stim_dir_super(
        stimuli_dir, objective, trial_type, row, image_type
    )
    batch_dir = os.path.join(stim_dir_super, f"batch_{batch_number}")

    return batch_dir


def make_dir_for_all_batches(stim_dir_super, n_batches_stop):
    """make directories specific to unit and batches"""

    for batch_number in range(n_batches_stop):
        # create folder specific to batch
        batch_dir = os.path.join(stim_dir_super, f"batch_{batch_number}")
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)


def get_tf_activations_list(model_instance, unit_specs_df):
    """create model_instance for each unique combinations of
    layer_name-pre_post_relu and return this list"""

    # create list of combinations of layer_name-pre_post_relu
    layer_str_list = []
    for idx, row in unit_specs_df.iterrows():
        layer_str = f"{row['layer_name']}_{row['pre_post_relu']}"
        layer_str_list.append(layer_str)

    # create model_instance for each unique combinations of layer_name-pre_post_relu
    tf_activations_list = []
    unique_layer_str_list = list(set(layer_str_list))
    unique_layer_str_list.sort()
    for cur_layer_str in unique_layer_str_list:
        tf_activations_list.append(model_instance(cur_layer_str))

    return tf_activations_list, unique_layer_str_list


def get_activation_according_to_objective(
    objective, activations_np, feature_map_number
):
    # get activations according to objective
    if objective == "neuron":
        # neuron number
        filter_location = activations_np.shape[2] // 2
        unit_activations = activations_np[
            :, filter_location, filter_location, feature_map_number
        ]  # batch_size, x, y, number_feature_maps
    elif objective == "channel":
        unit_activations = np.mean(np.mean(activations_np, axis=1), axis=1)[
            :, feature_map_number
        ]  # batch_size

    return unit_activations


def get_path_activations_whole_dataset_csv(
    stimuli_dir_load_from, objective_i, trial_type, cur_df
):
    stim_dir_super = os.path.join(
        stimuli_dir_load_from,
        objective_i,
        trial_type,
        f"layer_{cur_df['layer_number'].iloc[0]}",
    )
    activations_whole_dataset_csv = "activations_whole_dataset.csv"
    path_activations_whole_dataset_csv = os.path.join(
        stim_dir_super, activations_whole_dataset_csv
    )
    return path_activations_whole_dataset_csv


def write_unit_activations_to_csv(
    unit_activations, path_activations_whole_dataset_csv, paths, targets
):

    # convert torch tensor to list
    targets_list = targets.tolist()

    # if file does not exist yet, initialize csv and write
    if not os.path.isfile(path_activations_whole_dataset_csv):
        with open(path_activations_whole_dataset_csv, "w") as csvFile:
            csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
            csv_writer.writerow(["activations from whole batch"])
            csv_writer.writerow(["path to image", "activation", "target class"])
            for csv_row, path_i in enumerate(paths):
                csv_writer.writerow(
                    [path_i, unit_activations[csv_row].tolist(), targets_list[csv_row]]
                )
    # append to csv
    else:
        with open(path_activations_whole_dataset_csv, "a") as csvFile:
            csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
            for csv_row, path_i in enumerate(paths):
                csv_writer.writerow(
                    [path_i, unit_activations[csv_row].tolist(), targets_list[csv_row]]
                )


def save_activations_to_csv(
    batch_dir,
    val_or_train,
    layer_str,
    batch_size,
    paths,
    batch_number,
    feature_map_number,
    objective,
    indices_min,
    indices_max,
    unit_activations,
):
    """save activations to csv file"""

    csv_filename = "activations.csv"
    # write down what columns mean
    with open(os.path.join(batch_dir, csv_filename), "w") as csvFile:
        csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
        csv_writer.writerow(
            [
                str(val_or_train),
                layer_str,
                "batch_size_" + str(batch_size),
                paths[0].split("/n")[0],
            ]
        )
        csv_writer.writerow(
            [
                "batch_number: "
                + str(batch_number)
                + ", feature_map_number: "
                + str(feature_map_number)
                + ", objective:"
                + objective
            ]
        )
        csv_writer.writerow(
            ["batch_number", "idx_in_batch", "image_path", "activation"]
        )
    csvFile.close()

    # save activations to csv file
    with open(os.path.join(batch_dir, csv_filename), "a") as csvFile:  # 'a' for append
        csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
        # save idx_in_batch, path, activation
        for cur_list in [indices_min, indices_max]:
            for cur_ind in cur_list:
                csv_writer.writerow(
                    [
                        batch_number,
                        cur_ind,
                        paths[cur_ind]
                        .split("/" + str(val_or_train) + "/")[1]
                        .split(".JPEG")[0],
                        unit_activations[cur_ind],
                    ]
                )
    csvFile.close()


def save_natural_images_and_activations(
    batch_dir,
    objective,
    layer_str,
    images_np_transformed,
    indices_min,
    indices_max,
    unit_activations,
):
    """save the minimal and maximal natural images as well as activations"""

    # invert normalization and pick patch from images
    if objective == "neuron":
        # pick pixels according to receptive field size
        min_idx = 0
        max_idx = 10
        # from now on: batchsize, C, W, H
        images_np = invert_normalization(
            images_np_transformed[
                :, min_idx : max_idx + 1, min_idx : max_idx + 1, :
            ].transpose(0, 3, 1, 2)
        )
    elif objective == "channel":
        # from now on: batchsize, C, W, H
        images_np = invert_normalization(
            images_np_transformed[:, :, :, :].transpose(0, 3, 1, 2)
        )

    min_max_str_list = ["min", "max"]

    for min_max_ind, cur_list in enumerate([indices_min, indices_max]):
        for enumerate_ind, cur_list_ind in enumerate(cur_list):
            fig = plt.figure(figsize=(1, 1))
            ax = fig.add_subplot(111)

            plt.imshow(images_np[cur_list_ind])

            remove_white_margins(ax)

            image_name = f"{min_max_str_list[min_max_ind]}_{enumerate_ind}.png"
            image_path = os.path.join(batch_dir, image_name)
            plt.savefig(image_path, bbox_inches="tight", transparent=True, pad_inches=0)
            plt.close()

    np.save(os.path.join(batch_dir, f"indices_min.npy"), indices_min)
    np.save(os.path.join(batch_dir, f"indices_max.npy"), indices_max)
    np.save(os.path.join(batch_dir, f"unit_activations.npy"), unit_activations)


def invert_normalization(X):
    """invert normalization in preprocessing 
    applied to batches of images
    
     Args:
        X:      numpy array, dtype=float32.
                new: expected dimensions: BxCxWxH
                  
    Returns:
        image:  numpy array. dimensions: BxWxHxC
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = np.transpose(X.copy(), [0, 2, 3, 1])
    image *= std[None, None, None]
    image += mean[None, None, None]
    image = np.clip(image, 1e-8, 1 - 1e-8)

    return image


###### SPECIFIC TO OPTIMIZED STIMULI ######


def save_optimized_images_and_objectives(
    max_or_min, images, objectives, loss_additional_global, parent_stimulus_directory
):
    """save the minimal and maximal optimized images"""

    # iterate over min and max
    number_images = len(images)
    for img_idx in range(number_images):
        img = PIL.Image.fromarray(np.uint8(np.clip(images[img_idx] * 255, 0, 255)))
        image_name = f"{max_or_min}_{img_idx}.png"
        image_path = os.path.join(parent_stimulus_directory, image_name)
        print(image_path)
        img.save(image_path)

    np.save(
        os.path.join(parent_stimulus_directory, f"{max_or_min}_objective_values.npy"),
        objectives,
    )
    np.save(
        os.path.join(
            parent_stimulus_directory,
            f"{max_or_min}_additional_global_diversity_loss.npy",
        ),
        loss_additional_global,
    )
