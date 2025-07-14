# EEG-To-Text Model

This project builds upon the "Are EEG-To-Text models working?" implementation by NeuSpeech (NeuSpeech/EEG-To-Text) and leverages the base framework and execution pipeline from MikeWang (mikewang/eeg-to-text).

## Table of Contents

* [Overview](#overview)
* [Repository References](#repository-references)
* [Setting Up Your Conda Environment](#setting-up-your-conda-environment)
* [Dataset](#dataset)
  * [Downloading the Data](#downloading-the-data)
  * [Organizing the Data](#organizing-the-data)
* [Data Preparation](#data-preparation)
* [Batch Processing on Remote Clusters](#batch-processing-on-remote-clusters)

## Overview

This repository adapts and extends the NeuSpeech EEG-To-Text codebase. It provides:

* Scripts to convert raw MATLAB EEG recordings into pickle files.
* A standardized directory layout for dataset management.
* Example configurations for training and inference.

## Repository References

* **Implementation**: NeuSpeech/EEG-To-Text  
  [https://github.com/NeuSpeech/EEG-To-Text](https://github.com/NeuSpeech/EEG-To-Text)
* **Framework & Execution**: MikeWangWZHL/EEG-To-Text  
  [https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/README.md](https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/README.md)

## Setting Up Your Conda Environment

This project has been tested with **BART** and **T5** language models. Due to changes in library versions and compatibility issues, you need to select which model you plan to use and **uncomment the appropriate lines** in the `environment.yml` file before creating the environment.

1. Open the `environment.yml` file.
2. Uncomment the line(s) corresponding to the **transformers** version and model you want to use.
3. Then run the following command to create your environment:

```bash
conda env create -f environment.yml
```

## Dataset

The ZuCo datasets (v1 and v2) are **not** included in this repository due to their large file sizes. You must download them separately.

### Downloading the Data

1. **ZuCo v1** (Tasks 1 SR , 2 NR and 3 TSR)

   * Navigate to: [https://osf.io/q3zws/files/osfstorage](https://osf.io/q3zws/files/osfstorage)
2. **ZuCo v2** (Task 1 NR)

   * Navigate to: [https://osf.io/2urht/files/osfstorage](https://osf.io/2urht/files/osfstorage)

You can download the files via a web browser or use the `osfclient` for programmatic access in terminal:

```bash
pip install osfclient
export OSF_TOKEN=your_osf_token_here
# Clone v1 data
osf -p q3zws clone
# Clone v2 data
osf -p 2urht clone
```

### Organizing the Data

After downloading, move the MATLAB files into the following directory structure in your project:

```
~/dataset/ZuCo/
├── task1-SR/
│   └── Matlab_files/
├── task2-NR/
│   └── Matlab_files/
├── task3-TSR/
│   └── Matlab_files/
└── task1-NR-2.0/
    └── Matlab_files/
```


## Data Preparation

Convert the raw MATLAB EEG recordings into pickle format by running:

```bash
bash ./scripts/prepare_dataset.sh
```

This script calls:
```bash
python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task1-SR
python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task2-NR
python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task3-TSR
python3 ./util/construct_dataset_mat_to_pickle_v2.py
```
you will find that for v2 the line is commented and what is excuted is:

```bash
python3 ./util/construct_dataset_mat_to_pickle_memory.py
```

which handles large datasets robustly by resuming from the last processed file in case of interruptions.


## Batch Processing on Remote Clusters

If your interactive node has time or resource limits, consider the following:

1. Modify the shell scripts (`.sh`) to include resource directives (e.g., for SLURM):

 ```bash

\#!/bin/bash
#SBATCH --job-name=prepare\_dataset
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu                    
#SBATCH --gres=gpu:2  
```
and then in your terminal run 

 ```bash
sbash ./scripts/prepare_dataset.sh
```

Refer to your documentation and needs this is just an example of resource allocation.

2. The `construct_dataset_mat_to_pickle_memory.py` script will continue processing from the last saved state after any termination, making it suitable for batch jobs.


