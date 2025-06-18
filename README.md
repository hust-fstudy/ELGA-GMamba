# ELGA-GMamba: An Efficient Local-to-Global Awareness Graph Mamba for Object Recognition With Event-Based Cameras

The code will be released after the paper is published.

## Overview

![framework](./assets/framework.svg)

## Installation

### Requirements

All the codes are tested in the following environment:

- Linux (Ubuntu 20.04)
- Python 3.12
- PyTorch 2.4.0
- CUDA 11.8

### Dataset Preparation

All datasets should be downloaded and placed within the `dataset` directory, adhering to the folder naming rules and structure specified for the `N-Caltech101` and `DvsGesture` datasets as provided in the project.

## Quick Start

Clone the repository to your local machine:

```
git clone https://github.com/hust-fstudy/ELGA-GMamba
cd ELGA-GMamba
```

Once the dataset is specified in the `dataset_dict` dictionary within the `main` function of the `run_recognition.py` file, we can train and test it using the following command:

```bash
python run_recognition.py
```

