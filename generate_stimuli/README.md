# Generate stimuli for psychophysical experiment
# General
Before you run the code, you have to modify the path to ImageNet set in `natural_stimuli_save_all_activations.py`.
Also, you might have to modify some paths in `save_natural_stimuli.py`.
You have to (un)comment a few constants in `save_natural_stimuli.py` to either run the code for Experiment I or Experiment II.

## Experiment I
1. Run `generate_layer_folder_mapping_csv_experiment_i.py` (fast)
2. Run `generate_layer_folder_mapping_csv_for_all_feature_maps.py` (fast)
3. Run `natural_stimuli_save_all_activations.py` (slow)
4. Run `save_natural_stimuli.py` to save the **natural** stimuli (slow, needs to be modified))
5. Run `save_optimized_stimuli.py` to save the **optimized** stimuli (slow, needs to be modified))

## Experiment II
1. Run `generate_layer_folder_mapping_csv_experiment_ii.py` (fast)
2. Run `generate_layer_folder_mapping_csv_for_all_feature_maps.py` (fast, can be skipped if already done for Experiment I)
3. Run `natural_stimuli_save_all_activations.py` (slow, can be skipped if already done for Experiment I)
4. Run `save_natural_stimuli.py` to save the **natural** stimuli (slow, needs to be modified)
5. Run `save_optimized_stimuli.py` to save the **optimized** stimuli (slow, needs to be modified)