from collections import OrderedDict


def get_kernel_size(feature_map_number, layer_str):
    """return kernel size that corresponds to feature_map_number
    
    Args:
        feature_map_number: int
        layer_str:          layer in InceptionV1 [str]
    
    Returns:
        kernel_size_str:    str with kernel size description    
    """

    if "_pre_relu" in layer_str:
        layer_str = layer_str.replace("_pre_relu", "")

    if (
        feature_map_number
        > idx_of_kernel_sizes_in_each_layer_dict[layer_str + "_" + kernel_size_list[-1]]
    ):
        kernel_size_str = -1
        print(
            "The value of feature_map_number is too high! It is "
            + str(feature_map_number)
            + ". However, there are only "
            + str(
                idx_of_kernel_sizes_in_each_layer_dict[
                    layer_str + "_" + kernel_size_list[-1]
                ]
            )
            + " channels in layer "
            + layer_str
        )

    for kernel_size_i in reversed(kernel_size_list):
        if (
            feature_map_number
            < idx_of_kernel_sizes_in_each_layer_dict[layer_str + "_" + kernel_size_i]
        ):
            kernel_size_str = kernel_size_i

    return kernel_size_str


def get_channel_number(feature_map_number, kernel_size_str, layer_str):
    """Return channel number that is specific to a kernel size.
    
    Args:
        feature_map_number: number of feature map, where the range goes over all kernel sizes jointly [int]
        layer_str:          layer in InceptionV1 [str]
        kernel_size_str:    str with kernel size description
    
    Returns:   
        channel_number_str: number of channel, where the range is specific to a kernel size [str]    
    """

    if kernel_size_str == "1x1":
        channel_number_str = feature_map_number
    else:
        n_channels_to_subtract = idx_of_kernel_sizes_in_each_layer_dict[
            layer_str
            + "_"
            + kernel_size_list[kernel_size_list.index(kernel_size_str) - 1]
        ]
        channel_number_str = feature_map_number - n_channels_to_subtract

    return str(channel_number_str)


layer_names_experiment_i = [
    # 'conv2d0',
    # 'conv2d1',
    # 'conv2d2',
    "mixed3a",
    "mixed3b",
    "mixed4a",
    "mixed4b",
    "mixed4c",
    "mixed4d",
    "mixed4e",
    "mixed5a",
    "mixed5b",
]

# skip the non-inception-layers & skip every second layer in experiment II
layer_names_experiment_ii = [
    # 'conv2d0',
    # 'conv2d1',
    # 'conv2d2',
    "mixed3a",
    # "mixed3b",
    "mixed4a",
    # "mixed4b",
    "mixed4c",
    # "mixed4d",
    "mixed4e",
    # "mixed5a",
    "mixed5b",
]


kernel_size_list = ["1x1", "3x3", "5x5", "pool"]

# This is according to the InceptionV1 implementation in lucid 0.3.8.
# There is a mistake in channel mixed4a...
idx_of_kernel_sizes_in_each_layer_dict = OrderedDict()
idx_of_kernel_sizes_in_each_layer_dict["mixed3a_1x1"] = 64
idx_of_kernel_sizes_in_each_layer_dict["mixed3a_3x3"] = 192
idx_of_kernel_sizes_in_each_layer_dict["mixed3a_5x5"] = 224
idx_of_kernel_sizes_in_each_layer_dict["mixed3a_pool"] = 256
idx_of_kernel_sizes_in_each_layer_dict["mixed3b_1x1"] = 128
idx_of_kernel_sizes_in_each_layer_dict["mixed3b_3x3"] = 320
idx_of_kernel_sizes_in_each_layer_dict["mixed3b_5x5"] = 416
idx_of_kernel_sizes_in_each_layer_dict["mixed3b_pool"] = 480
idx_of_kernel_sizes_in_each_layer_dict["mixed4a_1x1"] = 192
idx_of_kernel_sizes_in_each_layer_dict["mixed4a_3x3"] = 396
idx_of_kernel_sizes_in_each_layer_dict["mixed4a_5x5"] = 444
idx_of_kernel_sizes_in_each_layer_dict[
    "mixed4a_pool"
] = 508  # this one should be 512 according to Szegedy et al. 2014
idx_of_kernel_sizes_in_each_layer_dict["mixed4b_1x1"] = 160
idx_of_kernel_sizes_in_each_layer_dict["mixed4b_3x3"] = 384
idx_of_kernel_sizes_in_each_layer_dict["mixed4b_5x5"] = 448
idx_of_kernel_sizes_in_each_layer_dict["mixed4b_pool"] = 512
idx_of_kernel_sizes_in_each_layer_dict["mixed4c_1x1"] = 128
idx_of_kernel_sizes_in_each_layer_dict["mixed4c_3x3"] = 384
idx_of_kernel_sizes_in_each_layer_dict["mixed4c_5x5"] = 448
idx_of_kernel_sizes_in_each_layer_dict["mixed4c_pool"] = 512
idx_of_kernel_sizes_in_each_layer_dict["mixed4d_1x1"] = 112
idx_of_kernel_sizes_in_each_layer_dict["mixed4d_3x3"] = 400
idx_of_kernel_sizes_in_each_layer_dict["mixed4d_5x5"] = 464
idx_of_kernel_sizes_in_each_layer_dict["mixed4d_pool"] = 528
idx_of_kernel_sizes_in_each_layer_dict["mixed4e_1x1"] = 256
idx_of_kernel_sizes_in_each_layer_dict["mixed4e_3x3"] = 576
idx_of_kernel_sizes_in_each_layer_dict["mixed4e_5x5"] = 704
idx_of_kernel_sizes_in_each_layer_dict["mixed4e_pool"] = 832
idx_of_kernel_sizes_in_each_layer_dict["mixed5a_1x1"] = 256
idx_of_kernel_sizes_in_each_layer_dict["mixed5a_3x3"] = 576
idx_of_kernel_sizes_in_each_layer_dict["mixed5a_5x5"] = 704
idx_of_kernel_sizes_in_each_layer_dict["mixed5a_pool"] = 832
idx_of_kernel_sizes_in_each_layer_dict["mixed5b_1x1"] = 384
idx_of_kernel_sizes_in_each_layer_dict["mixed5b_3x3"] = 768
idx_of_kernel_sizes_in_each_layer_dict["mixed5b_5x5"] = 896
idx_of_kernel_sizes_in_each_layer_dict["mixed5b_pool"] = 1024


# These are the feature_map_numbers of feature_maps that Olah et al. considered
# intuitive in their appendix and the ones that we considered most interpretable.

feature_maps_judged_most_interpretable_warm_up = OrderedDict()
feature_maps_judged_most_interpretable_warm_up["mixed3a"] = [43]
feature_maps_judged_most_interpretable_warm_up["mixed4b"] = [504]
feature_maps_judged_most_interpretable_warm_up["mixed5b"] = [17]

feature_maps_judged_most_interpretable_sampled_trials = OrderedDict()
feature_maps_judged_most_interpretable_sampled_trials["mixed3a"] = [230]
feature_maps_judged_most_interpretable_sampled_trials["mixed3b"] = [462]
feature_maps_judged_most_interpretable_sampled_trials["mixed4a"] = [501]
feature_maps_judged_most_interpretable_sampled_trials["mixed4b"] = [465]
feature_maps_judged_most_interpretable_sampled_trials["mixed4c"] = [449]
feature_maps_judged_most_interpretable_sampled_trials["mixed4d"] = [516]
feature_maps_judged_most_interpretable_sampled_trials["mixed4e"] = [809]
feature_maps_judged_most_interpretable_sampled_trials["mixed5a"] = [720]
feature_maps_judged_most_interpretable_sampled_trials["mixed5b"] = [946]
