# save the most/least activating natural images as natural stimuli
# uses the output of the generate_layer_folder_mapping_csv_for_all_feature_maps script

import numpy as np
import pandas as pd
import os
import ast
from tqdm.auto import tqdm
from torch.utils import data
from torchvision.datasets.folder import default_loader as default_image_loader
from torchvision import transforms
import csv


batch_size = 10
n_batches = 1
# can be "warm_up" or "sampled" or "practice_and_explanation"
trial_mode = "practice_and_explanation"

base_dir = "stimuli_analysis_all_feature_maps_and_class"

# for experiment II
block_index = "a"
feature_maps_file_name = f"stimuli_experiment_ii/batch_block_{block_index}/layer_folder_mapping_{trial_mode}_trials.csv"
base_output_dir = f"stimuli_experiment_ii/batch_block_{block_index}/"

# for experiment i
# feature_maps_file_name = f"stimuli_experiment_i/layer_folder_mapping_{trial_mode}_trials.csv"
# base_output_dir = f"stimuli_experiment_i/"


print("Trial Mode:", trial_mode)
print("Batch Size:", batch_size)
print("#Batches:", n_batches)
print("Output dir:", base_output_dir)
print("Feature Map CSV:", feature_maps_file_name)


class ImageFileListDataSet(data.Dataset):
    def __init__(self, file_list, transform=None, target_transform=None):
        self.file_list = file_list
        self.transform = transform
        self.target_transform = target_transform

        self.loader = default_image_loader

    def __getitem__(self, index):
        impath = self.file_list[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.file_list)


neuron_base_output_dir = os.path.join(base_output_dir, "neuron", f"{trial_mode}_trials")
channel_base_output_dir = os.path.join(
    base_output_dir, "channel", f"{trial_mode}_trials"
)

neuron_csv_base_dir = os.path.join(base_dir, "neuron", "sampled_trials")
channel_csv_base_dir = os.path.join(base_dir, "channel", "sampled_trials")


feature_maps_df = pd.read_csv(feature_maps_file_name, header=1)


csv_min_filename = "activation_min.csv"
csv_max_filename = "activation_max.csv"


def get_randomized_indices(batch_size, n_batches, seed):
    """generate randomized order of indices such that the 20 
    images that belong to one bin (e.g. min_0.png) is different:
    randomize(0...20), then randomize(20...40) ... randomize(180...200)"""
    randomized_indices = np.empty([batch_size * n_batches])
    array_of_n_batches = np.arange(n_batches)

    np.random.seed(seed)
    for batch_i in range(batch_size):
        randomized_indices[
            (n_batches * batch_i) : (n_batches + batch_i * n_batches)
        ] = np.random.permutation(array_of_n_batches + batch_i * n_batches)

    return randomized_indices


def process_layer(
    raw_layer_number,
    layer_number,
    layer,
    csv_base_dir,
    output_base_dir,
    trial_mode,
    get_rf_size=lambda l, f: (0, 223),
):
    print("Loading DF...")
    input_csv_filename = os.path.join(
        csv_base_dir, f"layer_{layer_number}", "activations_whole_dataset.csv"
    )
    df = pd.read_csv(
        input_csv_filename, header=1, converters={"activation": ast.literal_eval}
    )

    print("DF loaded")

    for _, row in tqdm(
        feature_maps_df[feature_maps_df["layer_number"] == raw_layer_number].iterrows(),
        position=0,
        total=len(feature_maps_df[feature_maps_df["layer_number"] == raw_layer_number]),
    ):
        kernel_size = row["kernel_size_number"]
        channel = row["channel_number"]
        feature_map = row["feature_map_number"]
        layer_name = row["layer_name"]
        print(
            f"layer_name {layer_name}, feature_map {feature_map}, channel {channel}, kernel_size {kernel_size}"
        )

        df_expanded = df.copy()
        df_expanded["selected_activation"] = df["activation"].apply(
            lambda x: x[feature_map]
        )
        df_expanded_sorted = df_expanded.sort_values(
            "selected_activation", ascending=True
        )

        # create dataframes with relevant columns and rows only. Also, randomize the order in one image bin
        min_indices = get_randomized_indices(batch_size, n_batches, seed=feature_map)
        max_indices = get_randomized_indices(
            batch_size, n_batches, seed=feature_map + 1
        )
        min_images_activations_df = (
            df_expanded_sorted[: batch_size * n_batches]
            .drop(["activation", "target class"], axis=1)
            .iloc[min_indices]
        )
        max_images_activations_df = (
            df_expanded_sorted[-batch_size * n_batches :]
            .drop(["activation", "target class"], axis=1)
            .iloc[max_indices]
        )

        min_file_names = min_images_activations_df["path to image"].tolist()
        max_file_names = max_images_activations_df["path to image"].tolist()

        center_crop_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

        max_dataset = ImageFileListDataSet(
            max_file_names, transform=center_crop_transform
        )
        min_dataset = ImageFileListDataSet(
            min_file_names, transform=center_crop_transform
        )

        for image_idx_in_batch in tqdm(range(batch_size), position=1, leave=False):
            for batch in tqdm(range(n_batches), position=2, leave=False):
                if n_batches == 1:
                    image_idx = image_idx_in_batch
                elif trial_mode == "sampled":
                    image_idx = batch + batch_size * image_idx_in_batch

                max_image = max_dataset[image_idx]
                min_image = min_dataset[image_idx]

                min_idx, max_idx = get_rf_size(layer_name, feature_map)
                box = (min_idx, min_idx, max_idx + 1, max_idx + 1)

                max_image = max_image.crop(box)
                min_image = min_image.crop(box)

                if n_batches == 1:
                    output_dir = os.path.join(
                        output_base_dir,
                        f"layer_{raw_layer_number}",
                        f"kernel_size_{kernel_size}",
                        f"channel_{channel}",
                        "natural_images",
                    )
                elif trial_mode == "sampled":
                    output_dir = os.path.join(
                        output_base_dir,
                        f"layer_{raw_layer_number}",
                        f"kernel_size_{kernel_size}",
                        f"channel_{channel}",
                        "natural_images",
                        f"batch_{batch}",
                    )
                os.makedirs(output_dir, exist_ok=True)
                max_filename = os.path.join(output_dir, f"max_{image_idx_in_batch}.png")
                min_filename = os.path.join(output_dir, f"min_{image_idx_in_batch}.png")

                max_image.save(max_filename)
                min_image.save(min_filename)

                # save activation to csv
                if image_idx_in_batch == 0:
                    with open(
                        os.path.join(output_dir, csv_min_filename), "w"
                    ) as csvFile:
                        csv_writer = csv.writer(
                            csvFile, delimiter=",", lineterminator="\n"
                        )
                        csv_writer.writerow(["image_path", "idx", "activation"])
                    csvFile.close()
                    with open(
                        os.path.join(output_dir, csv_max_filename), "w"
                    ) as csvFile:
                        csv_writer = csv.writer(
                            csvFile, delimiter=",", lineterminator="\n"
                        )
                        csv_writer.writerow(["image_path", "idx", "activation"])
                    csvFile.close()

                with open(os.path.join(output_dir, csv_min_filename), "a") as csvFile:
                    csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
                    csv_writer.writerow(
                        [
                            min_images_activations_df.iloc[image_idx, 0],
                            image_idx_in_batch,
                            min_images_activations_df.iloc[image_idx, 1],
                        ]
                    )
                csvFile.close()

                with open(os.path.join(output_dir, csv_max_filename), "a") as csvFile:
                    csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
                    csv_writer.writerow(
                        [
                            max_images_activations_df.iloc[image_idx, 0],
                            image_idx_in_batch,
                            max_images_activations_df.iloc[image_idx, 1],
                        ]
                    )
                csvFile.close()


alphabet_layer_dict = {
    "a": "0",
    "b": "2",
    "c": "7",
    "d": "1",
    "e": "2",
    "f": "4",
    "g": "6",
    "h": "8",
    "i": "3",
    "j": "8",
    "k": "0",
    "l": "7",
    "m": "1",
    "n": "3",
    "o": "0",
    "p": "6",
    "q": "2",
    "r": "5",
    "s": "6",
    "t": "7",
    "u": "2",
    "v": "1",
    "w": "3",
}


layers = feature_maps_df.layer_name.unique()
raw_layer_numbers = [
    feature_maps_df.loc[feature_maps_df.layer_name == l, "layer_number"].tolist()[0]
    for l in layers
]

for layer in layers:
    print("Layer:", layer)

layer_numbers = [
    l if isinstance(l, int) or l.isdigit() else alphabet_layer_dict[l]
    for l in raw_layer_numbers
]

print("Layers:", layers)
print("Layer Numbers:", layer_numbers)

for raw_layer_number, layer_number, layer in zip(
    raw_layer_numbers, layer_numbers, layers
):
    print("Layer:", layer, "with number:", layer_number)
    print("Writing images for channel objective...")
    process_layer(
        raw_layer_number,
        layer_number,
        layer,
        channel_csv_base_dir,
        channel_base_output_dir,
        trial_mode,
    )
