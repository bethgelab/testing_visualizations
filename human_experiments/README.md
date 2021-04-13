# Run psychophysical experiments

# General
In total, we ran two different experiments. 
Make sure that you have all the necessary requirements installed. [requirements.txt](requirements.txt) works for both experiments.
Further, set the correct paths to your stimulus folder in `config.py`; for the folder structure and feature map choices see below.
Then, you only have to run `main.py`.

Note that the code for the two experiments is very similar as we developed Experiment II after Experiment I. 
Sometimes, we refer to the two different experiments as the "main" (Experiment I) and the "ablation" (Experiment II) experiments.

## Experiment I
For the feature map choices, see [exp_i/layer_folder_mapping](exp_i/layer_folder_mapping).

The code assumes that the stimuli are stored in the following structure: 
```
parent_folder
│
└───channel
    │
    └───sampled_trials
    │   │
    │   └───layer_0
    │   │   │
    │   │   └───kernel_size_0
    │   │   │   │
    │   │   │   └───channel_0
    │   │   │       │
    │   │   │       └───natural_images
    │   │   │       │   │
    │   │   │       │   └───batch_[0-19]
    │   │   │       │           [min,max]_[0-9].png
    │   │   │       │
    │   │   │       └───optimized_images
    │   │   │                   [min,max]_[0-9].png
    │   │   │
    │   │   └───kernel_size_[1-2] as for kernel_size_0
    │   │   │    
    │   │   └───kernel_size_3
    │   │       │
    │   │       └───channel_0 (as above)
    │   │       │
    │   │       └───channel_1 (structure as above, but corresponds to hand-picked unit!)
    │   │   
    │   └───layer_[1-9, i-w] as for layer_0
    │
    └───warm_up_trials
        │
        └───layer_a
        │   │
        │   └───kernel_size_a
        │       │
        │       └───channel_a
        │           │
        │           └───natural_images
        │           │       [min,max]_[0-9].png
        │           │
        │           └───optimized_images
        │                   [min,max]_[0-9].png
        │
        └───layer_[b-c] as for layer_a
```

## Experiment II
For the feature map choices, see [exp_ii/layer_folder_mapping](exp_ii/layer_folder_mapping).

The code assumes that the stimuli are stored in the following structure: 

```
parent_folder
│
└───batch_block[a-h]
│   │
│   └───channel
│       │
│       └───sampled_trials
│           │
│           └───layer_0
│           │   │
│           │   └───kernel_size_0
│           │   │   │
│           │   │   └───channel_0
│           │   │       │
│           │   │       └───natural_images
│           │   │       │   │
│           │   │       │   └───batch_[0-19]
│           │   │       │           [min,max]_[0-9].png
│           │   │       │
│           │   │       └───optimized_images
│           │   │                   [min,max]_[0-9].png
│           │   │
│           │   └───kernel_size_[1-3] as for kernel_size_0
│           │   
│           └───layer_[2, 4, 6, 8] as for layer_0
│       
└───warm_up_trials
    │
    └───layer_a
    │   │
    │   └───kernel_size_a
    │       │
    │       └───channel_a
    │           │
    │           └───natural_images
    │           │       [min,max]_[0-9].png
    │           │
    │           └───optimized_images
    │                   [min,max]_[0-9].png
    │
    └───layer_[b-c] as for layer_a
```













