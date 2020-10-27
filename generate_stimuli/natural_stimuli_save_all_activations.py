# Generate Natural Stimuli - for ALL feature maps
# ---all feature maps and target class saved in one file---

import numpy as np
import random
import tensorflow as tf
import os
import pandas as pd
import lucid.modelzoo.vision_models as models
from render import import_model
import utils_stimulus_generation as ut_stim
import time
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# only channel objective was used in published experiments
objective_list = ["channel", "neuron"] 
n_batches_to_generate_stimuli = 159
stimuli_dir = "stimuli_analysis_all_feature_maps_and_class"
trial_type = "sampled_trials"


# Load model
# import InceptionV1 from the Lucid modelzoo
model = models.InceptionV1()
model.load_graphdef()


image_type = "natural"
tf.set_random_seed(1234)
random.seed(0)
np.random.seed(0)
val_or_train = "val"


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


# calculate the batch_size based on the size of the data set and
# the number of batches you would like to get stimuli for
if val_or_train == "val":
    n_images_total_in_ImageNet = 50000
elif val_or_train == "train":
    n_images_total_in_ImageNet = 1281167
batch_size = (
    n_images_total_in_ImageNet // n_batches_to_generate_stimuli
)  # always rounds down
# correct batch size
# if it is too large (long runtime), set it to 256
if batch_size > 256:
    batch_size = 256
# if it is too small (i.e. min and max stimuli would overlap), raise exception
elif batch_size < ut_stim.number_images:
    raise Exception(
        "The batch_size is <18. You wanted to generate images for too many patches "
        "for the chosen data set. Please lower the number of patches and/or choose "
        " a different data set."
    )
print("This is the batch_size: ", batch_size)


# import ImageNet
datapath = ...  # TODO

# get data
data_dir = os.path.join(datapath, val_or_train)

# make deterministic
torch.manual_seed(1234)

# preprocessing (corresponds to ResNet)
this_dataset = ImageFolderWithPaths(
    data_dir,
    transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    ),
)

data_loader = torch.utils.data.DataLoader(
    this_dataset, batch_size=batch_size, shuffle=False, num_workers=70, pin_memory=True,
)


# read in unit specifications from csv into pandas dataframe
path_to_csv_file = os.path.join(stimuli_dir, f"layer_folder_mapping_{trial_type}.csv")
unit_specs_df = pd.read_csv(path_to_csv_file, header=1)

# for ALL feature maps
# make directories until layer-level
for objective_i in objective_list:
    for layer_number in range(10):
        cur_dir = os.path.join(
            stimuli_dir, objective_i, "sampled_trials", f"layer_{layer_number}"
        )
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

# for all but last batch
# for ALL feature maps
# different structure: csv saves list of values
# for whole dataset: save relevant (according to neuron or channel objective)
# activation to csv
with tf.Graph().as_default() as graph, tf.Session() as sess:

    image = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
    model_instance = import_model(model, image)
    tf_activations_list, unique_layer_str_list = ut_stim.get_tf_activations_list(
        model_instance, unit_specs_df
    )

    # loop through batches
    for batch_number, (images, targets, paths) in enumerate(data_loader):
        start_time = time.time()
        print("batch_number", batch_number)
        if batch_number == len(data_loader) - 1:
            last_batchs_batch_size = images.shape[0]
            print(f"breaking at {batch_number}")
            break

        # forward pass
        images_np_transformed = images.numpy().transpose(0, 2, 3, 1)
        activations_list = sess.run(
            tf_activations_list, {image: images_np_transformed}
        )  # batch_size, x, y, number_feature_maps

        # save it!
        # loop through layers
        for layer_idx, cur_layer_str in enumerate(unique_layer_str_list):
            activations_np = activations_list[layer_idx]

            # loop through objectives
            for objective_i in objective_list:
                unit_activations = ut_stim.get_activation_according_to_objective(
                    objective_i, activations_np, np.arange(0, activations_np.shape[-1])
                )
                # create folder to save csv in
                activations_whole_dataset_csv = "activations_whole_dataset.csv"
                layer_dir = os.path.join(
                    stimuli_dir, objective_i, "sampled_trials", f"layer_{layer_idx}"
                )
                path_activations_whole_dataset_csv = os.path.join(
                    layer_dir, activations_whole_dataset_csv
                )
                # write activation to csv
                ut_stim.write_unit_activations_to_csv(
                    unit_activations, path_activations_whole_dataset_csv, paths, targets
                )

        end_time = time.time()
        print("time elapsed ", end_time - start_time)


# for all but last batch
# for ALL feature maps
# different structure: csv saves list of values
# for whole dataset: save relevant (according to neuron or channel objective)
# activation to csv
with tf.Graph().as_default() as graph, tf.Session() as sess:

    image = tf.placeholder(tf.float32, shape=(80, 224, 224, 3))
    model_instance = import_model(model, image)
    tf_activations_list, unique_layer_str_list = ut_stim.get_tf_activations_list(
        model_instance, unit_specs_df
    )

    # loop through batches
    for batch_number, (images, targets, paths) in enumerate(data_loader):
        start_time = time.time()

        if batch_number == len(data_loader) - 1:
            print(f"actually running batch_number {batch_number}")
            # forward pass
            images_np_transformed = images.numpy().transpose(0, 2, 3, 1)
            activations_list = sess.run(
                tf_activations_list, {image: images_np_transformed}
            )  # batch_size, x, y, number_feature_maps

            # save it!
            # loop through layers
            for layer_idx, cur_layer_str in enumerate(unique_layer_str_list):
                activations_np = activations_list[layer_idx]

                # loop through objectives
                for objective_i in objective_list:
                    unit_activations = ut_stim.get_activation_according_to_objective(
                        objective_i,
                        activations_np,
                        np.arange(0, activations_np.shape[-1]),
                    )
                    # create folder to save csv in
                    activations_whole_dataset_csv = "activations_whole_dataset.csv"
                    layer_dir = os.path.join(
                        stimuli_dir, objective_i, "sampled_trials", f"layer_{layer_idx}"
                    )
                    path_activations_whole_dataset_csv = os.path.join(
                        layer_dir, activations_whole_dataset_csv
                    )
                    # write activation to csv
                    ut_stim.write_unit_activations_to_csv(
                        unit_activations,
                        path_activations_whole_dataset_csv,
                        paths,
                        targets,
                    )
        else:
            print(f"skipped batch_number {batch_number}")

        end_time = time.time()
        print("\ttime elapsed ", end_time - start_time)
