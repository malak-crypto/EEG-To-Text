# EEG-To-Text Model

This project builds upon the "Are EEG-To-Text models working?" implementation by NeuSpeech (NeuSpeech/EEG-To-Text) and leverages the base framework and execution pipeline from MikeWangWZHL (mikewang/eeg-to-text).

## Table of Contents

* [Overview](#overview)
* [Repository References](#repository-references)
* [Dataset](#dataset)

  * [Downloading the Data](#downloading-the-data)
  * [Organizing the Data](#organizing-the-data)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Usage](#usage)
* [Batch Processing on Remote Clusters](#batch-processing-on-remote-clusters)
* [Contributing](#contributing)
* [License](#license)

## Overview

This repository adapts and extends the NeuSpeech EEG-To-Text codebase to facilitate training and evaluation on the ZuCo (Zurich Cognitive Language Processing) datasets. It provides:

* Scripts to convert raw MATLAB EEG recordings into pickle files.
* A standardized directory layout for dataset management.
* Example configurations for training and inference.

## Repository References

* **Original Implementation**: NeuSpeech/EEG-To-Text<br>
  [https://github.com/NeuSpeech/EEG-To-Text](https://github.com/NeuSpeech/EEG-To-Text)
* **Framework & Execution**: MikeWangWZHL/EEG-To-Text<br>
  [https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/README.md](https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/README.md)

## Dataset

The ZuCo datasets (v1 and v2) are **not** included in this repository due to their large file sizes. You must download them separately.

### Downloading the Data

1. **ZuCo v1** (Tasks 1 SR and 2 NR)

   * Navigate to: [https://osf.io/q3zws/files/osfstorage](https://osf.io/q3zws/files/osfstorage)
2. **ZuCo v2** (Task 1 NR)

   * Navigate to: [https://osf.io/2urht/files/osfstorage](https://osf.io/2urht/files/osfstorage)

You can download the files via a web browser or use the `osfclient` for programmatic access:

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
└── task1-NR-2.0/
    └── Matlab_files/
```

## Installation

1. Clone this repository:

   ```bash
   ```

git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName

````

2. Install Python dependencies:

   ```bash
pip install -r requirements.txt
````

## Data Preparation

Convert the raw MATLAB EEG recordings into pickle format by running:

```bash
bash ./scripts/prepare_dataset.sh
```

This script calls:

```bash
python3 ./util/construct_dataset_mat_to_pickle_memory.py
```

which handles large datasets robustly by resuming from the last processed file in case of interruptions.

## Usage

1. **Training**:

   ```bash
   ```

python train.py --config configs/train\_config.yaml

````

2. **Evaluation/Inference**:

   ```bash
python inference.py --checkpoint path/to/checkpoint.pth \
                   --data-dir ~/dataset/ZuCo/
````

Refer to the `configs/` directory for detailed options and hyperparameters.

## Batch Processing on Remote Clusters

If your interactive node has time or resource limits, consider the following:

1. Modify the shell scripts (`.sh`) to include resource directives (e.g., for SLURM):

   ```bash
   ```

\#!/bin/bash
\#SBATCH --job-name=prepare\_dataset
\#SBATCH --time=04:00:00
\#SBATCH --mem=32G
\#SBATCH --cpus-per-task=8
bash ./scripts/prepare\_dataset.sh

```

2. The `construct_dataset_mat_to_pickle_memory.py` script will continue processing from the last saved state after any termination, making it suitable for batch jobs.

## Contributing

Contributions are welcome! Please fork the repository and open a pull request with your changes. Ensure that dataset paths and configurations are clearly documented.

## License

This project inherits the license of the original repositories. Please review:

- [NeuSpeech/EEG-To-Text LICENSE](https://github.com/NeuSpeech/EEG-To-Text/blob/main/LICENSE)
- [MikeWangWZHL/EEG-To-Text LICENSE](https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/LICENSE)

---

*Happy coding!*

```

