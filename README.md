# Highway Environment with LLM-based Dataset Generation

This repository provides a simulation environment using `gym` and `highway-env` to model highway driving scenarios. The agent interacts with the environment, and Long Language Model (LLM)-based prompts can be used to recommend actions for dataset generation. This README file includes detailed instructions on how to set up the environment and install all required dependencies.

## Table of Contents
- [Installation](#installation)
  - [Installing Required Libraries](#installing-required-libraries)
- [Generating Datasets with LLM](#generating-datasets-with-llm)

## Installation

### Installing Required Libraries

All necessary dependencies are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

## Generating Datasets with LLM

### generating-datasets-with-llm

Run the Dataset Generation Script:

Run the dataset generation script with your desired configurations:

```bash
python3 random_data_updated.py 
```
Modify Hyperparameters:

Adjust hyperparameters such as episodes, samples_per_episode, vehicleCount_range,

Check the Generated Dataset:

The generated dataset will be saved as a CSV file in the datasets directory. V
