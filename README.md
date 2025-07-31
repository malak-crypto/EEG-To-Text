## Table of Contents

* [Overview](#overview)
* [Repository References](#repository-references)
* [Setting Up Your Conda Environment](#setting-up-your-conda-environment)
* [Dataset](#dataset)

  * [Downloading the Data](#downloading-the-data)
  * [Organizing the Data](#organizing-the-data)
* [Data Preparation](#data-preparation)
* [Batch Processing on Remote Clusters](#batch-processing-on-remote-clusters)
* [Training & Evaluation](#training--evaluation)

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

1. **ZuCo v1** (Tasks 1 SR, 2 NR and 3 TSR)

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

For ZuCo v2, the default uses a memory-safe script:

```bash
python3 ./util/construct_dataset_mat_to_pickle_memory.py
```

This handles large datasets robustly by resuming from the last processed file in case of interruptions.

## Batch Processing on Remote Clusters

If your interactive node has time or resource limits, consider the following:

1. Modify the shell scripts (`.sh`) to include SLURM resource directives. Example:

```bash
#!/bin/bash
#SBATCH --job-name=prepare_dataset
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
```

2. Submit via:

```bash
sbatch ./scripts/prepare_dataset.sh
```

3. The `construct_dataset_mat_to_pickle_memory.py` script will continue processing from the last saved state after any termination, making it suitable for batch jobs.

## Training & Evaluation

All training and evaluation scripts are located in the `scripts/` directory. To run a job:

1. **Choose your script** from `./scripts/` (e.g., train_BART_1step.sh, train_T5_2step.sh, eval_BART_1step.sh , etc).
2. In your terminal, launch the job with:

   ```bash
   sbatch ./scripts/<name-of-script-you-chose>
   ```

### Evaluation Notes

* The main evaluation logic lives in `eval_decoding.py`.
* If you use **BART** or **T5**, open `eval_decoding.py` and locate the lines after:

  ```python
  gen_out = model.generate(...)
  ```

  You must **uncomment** the appropriate output-processing block for your model and **comment out** the other. This ensures the logits decoding matches BART or T5's expected format.

### CSCL (Contrastive Semantic-aware Learning)

We experimented with CSCL pre-encoder and adapter modules (`*_cscl.py` files), but found scaling issues:

1. Full-dataset CSCL pre-encoder training crashed mid-run.
2. Task1-only training succeeded but yielded subpar metrics.
3. Sequential pre-encoder (task1) + full-model training resulted in zero scores.
4. Adapter-based fine-tuning also produced zero scores.

As a result, all `*_cscl.py` scripts are **commented out** by default. Feel free to:

1. Uncomment any `*_cscl.py` file.
2. Adjust hyperparameters in the corresponding script.
3. Rerun via `sbatch ./scripts/<script>`.

Feedback and contributions are welcome!
