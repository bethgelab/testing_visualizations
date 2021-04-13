# Data

This folder contains the data in comma-separaterd csv format after a few preprocessing steps.

# Main, practice, catch, repeated trials

Files:
* [experiment_I_main_practice_catch_repeated_trials_preprocessed.csv](experiment_I_main_practice_catch_repeated_trials_preprocessed.csv)
* [experiment_II_main_practice_catch_trials_preprocessed.csv](experiment_II_main_practice_catch_trials_preprocessed.csv)

These were the **preprocessing steps**:
* add a column `unit_name` with the unit name of the format "layer, branch [int.]"
* add a column `condition_name` of the format "mode-n_instruction_images"
* add a column `abs_conf_rating` with the absolute confidence rating
* rename the column `subject_id` to `subject_id_within_expert_type`
    * (only for Exp II) in column `subject_id`, enter the id from `subject_id_within_expert_type` for experts, and enter the id from `subject_id_within_expert_type` + 10 for lay people
    * (only for Exp II) add a column with the expert level `expert_level`

Here are **descriptions of the columns**:
* `subject_id_within_expert_type`: original subject id when collecting data - for analysis purposes, stick to subject_id (int)
* `trial_nr`: number of trial in the experiment of one participant starting at 0 in practice trials and continuously increasing by one throughout main and catch (as well as repeated in the case of Experiment I) trials. (int)
* `block_nr`: number of block in the experiment of one participant starting at 0 with the practice none trials and continuously increasing. For orientation, the block number increases whenever the shading or color increases in Figure 8. (int)
* (only in Exp II) `batch_block`: name of folder that images were sampled from. (str)
* `instr_type`: condition of reference images: optimized (referring to synthetic) or natural (or in Exp: none_pre or none_post, referring to the none condition at the _beginning_ or _end_ of the experiment (str)
* `trial_type`: practice or main (or in Exp: repeated) (str)
* (only in Exp II) `instruction_grid_mode`: whether only maximally or minimally _and_ maximally activating images are shown (str)
* (only in Exp II) `n_instruction_patches`: how many instruction images are displayed on the sides (int)
* `catch_trial`: whether the current trial is a catch trial (bool)
* `batch`: name of folder that the natural query (and potentially the reference images in the natural condition) are taken from (int)
* `layer`: layer of which the feature map is shown from (str)
* `kernel_size`: kernel size of which the feature map is shown from (str)
* `channel`: channel of which the feature map is shown form; `channel_0` corresponds to the randomly sampled feature maps, while `channel_1` contains the hand-picked feature maps(str)
* `query_order`: order of query images (str)
* `correct`: whether trial was correctly answered (bool) 
* `conf_rating`: confidence rating with negative and positive values to distinguish the selected image (int)
* `RT`: reaction time (float)
* `timeout`: whehter participant took longer than given time. In fact, this was unused because we decided for no time restriction per trial. (bool)
* `response`: whether the upper or lower image was selected (str)
* `correct_response`: which response is correct (str)
* `query_image_a`: path to query image a (str)
* `query_image_b`: path to query image b (str)
* `old_trial_nr`: in case of repeated trials, the trial number when the participant saw this very trial the first time (float)
* `old_block_nr`: in case of repeated trials, the block number when the participant saw this very trial the first time (float)
* `seed`: seed saved from config (int)
* `unit_name`: name of the unit, see above (str)
* (only in Exp II) `condition_name`: name of the condition, see above (str)
* `abs_conf_rating`: absolute confidence rating (int)
* `subject_id`: artificial subject id across all participants within one experiment (int)
* (only in Exp II) `expert_level`: either "expert" or "naive" (str)


# Intuitiveness trials

Files:
* [experiment_I_intuitiveness_beginning_preprocessed.csv](experiment_I_intuitiveness_beginning_preprocessed.csv)
* [experiment_II_intuitiveness_beginning_preprocessed.csv](experiment_II_intuitiveness_beginning_preprocessed.csv)
* [experiment_II_intuitiveness_end_preprocessed.csv](experiment_II_intuitiveness_end_preprocessed.csv)

These were the **preprocessing steps**:
* add a column `trial_name_as_jittered_number` to jitter the x-values of single participant's ratings
* add a column `trial_name_number` with the trial name converted to a number
* add a column `subject_id_within_expert_type` to keep the original subject id when collecting data

Here are **descriptions of the columns**:
* `subject_id`: subject id across all participants within one experiment (in case of Experiment II, this is an artificially created one) (int)
* (only in Exp II) `subject_id_within_expert_type`: original subject id when collecting data - for analysis purposes, stick to subject_id (int)
* `trial_name`: name of trial, namely either `a`, `b`, or `c` (str)
* `trial_nr`: order in which trials were shown (int)
* `rating`: intuitiveness rating between -100 and 100 (int)
* `rt`: reaction time (float)
* (only in Exp II) `trial_name_number`: column `trial_name` converted to 0.0, 1.0, and 2.0 (float)
* `trial_name_as_jittered_number`: `trial_name_number` with added jitter for plotting purposes