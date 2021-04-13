# Generate Optimized Stimuli

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd
from tqdm import tqdm
import argparse
import time

# lucid imports
import lucid.modelzoo.vision_models as models
from lucid.optvis import transform
import lucid.optvis.param as param

# just modified render & objectives file and not the original files from lucid
import render
import objectives

# custom imports
import utils_stimulus_generation as ut_stim


# Load experiment specification
parser = argparse.ArgumentParser(description="Generate Optimized Stimuli")
parser.add_argument(
    "-d",
    "--stimuli_dir",
    required=True,
    help="folder where csv data is read in from and where stimuli are stored to",
)
parser.add_argument(
    "-o",
    "--objective",
    default="channel",
    help="choose images according to the channel objective",
)
parser.add_argument(
    "-n", "--n-images", default=10, type=int, help="how many images to generate"
)
parser.add_argument(
    "-t",
    "--trial_type",
    default="sampled_trials",
    help="which kind of trials the stimuli should be generated for: either sampled_trials or warm_up_trials or practice_and_explanation_trials",
)

args = parser.parse_args()
objective = args.objective
stimuli_dir = args.stimuli_dir
trial_type = args.trial_type
number_images = args.n_images

# Load model
# import a model from the Lucid modelzoo
model = models.InceptionV1()
model.load_graphdef()

# Parameters
image_type = "optimized"
tf.set_random_seed(1234)

# Produce minimally and maximally activating images
# read in unit specifications from csv into pandas dataframe
path_to_csv_file = os.path.join(stimuli_dir, f"layer_folder_mapping_{trial_type}.csv")
unit_specs_df = pd.read_csv(path_to_csv_file, header=1)

ut_stim.make_all_dirs(
    stimuli_dir, objective, f"{trial_type}_trials", image_type, unit_specs_df
)

def get_channel_objective_stimuli(layer, channel):
    img_size = 224

    padding_size = 16
    param_f = lambda: param.image(img_size + 2 * padding_size, batch=number_images)
    objective_per_image = objectives.channel(layer, channel)
    diversity_loss = -1e2 * objectives.diversity(layer)

    # transformations as described in Feature Visualization blog post
    kwargs = dict(
        thresholds=(2560,),
        optimizer=tf.train.AdamOptimizer(learning_rate=0.05),
        transforms=[
            transform.jitter(16),
            transform.random_scale((1.0, 0.975, 1.025, 0.95, 1.05)),
            transform.random_rotate((-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)),
            transform.jitter(8),
        ],
    )

    # generate min stimuli
    _, min_stimuli, min_loss, loss_additional_global_list = render.render_vis(
        model,
        -objective_per_image,
        diversity_loss,
        param_f,
        use_fixed_seed=True,
        **kwargs,
    )
    # the optimization saves multiple states of the results
    # the last item is the final value
    min_stimuli = min_stimuli[-1]
    min_loss = min_loss[-1]
    min_loss_additional_global = loss_additional_global_list[-1]

    # undo/crop padding
    min_stimuli = min_stimuli[:, padding_size:-padding_size, padding_size:-padding_size]

    # invert loss again
    min_loss = -min_loss

    # generate max stimuli
    _, max_stimuli, max_loss, loss_additional_global_list = render.render_vis(
        model,
        objective_per_image,
        diversity_loss,
        param_f,
        use_fixed_seed=True,
        **kwargs,
    )
    # see above
    max_stimuli = max_stimuli[-1]
    max_loss = max_loss[-1]
    max_loss_additional_global = loss_additional_global_list[-1]

    # undo/crop padding
    max_stimuli = max_stimuli[:, padding_size:-padding_size, padding_size:-padding_size]

    return (
        min_stimuli,
        min_loss,
        min_loss_additional_global,
        max_stimuli,
        max_loss,
        max_loss_additional_global,
    )


# loop through each row
for idx, cur_row in tqdm(unit_specs_df.iterrows(), total=len(unit_specs_df)):
    start_time = time.time()
    layer = f"{cur_row['layer_name']}_{cur_row['pre_post_relu']}"
    channel = cur_row["feature_map_number"]

    if objective == "channel":
        (
            min_stimuli,
            min_loss,
            min_loss_additional_global,
            max_stimuli,
            max_loss,
            max_loss_additional_global,
        ) = get_channel_objective_stimuli(layer, channel)
    else:
        raise ValueError("The objective must be channel.")

    # save images
    stim_dir_super = ut_stim.get_stim_dir_super(
        stimuli_dir, objective, f"{trial_type}_trials", cur_row, image_type
    )
    ut_stim.save_optimized_images_and_objectives(
        "min", min_stimuli, min_loss, min_loss_additional_global, stim_dir_super
    )
    ut_stim.save_optimized_images_and_objectives(
        "max", max_stimuli, max_loss, max_loss_additional_global, stim_dir_super
    )
    print(f"layer: {layer}, channel: {channel}, time = {time.time() - start_time:.1f}s")