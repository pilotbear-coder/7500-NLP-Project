# Data Directory

This folder contains documentation regarding the datasets used in this project. 

> **Note:** Raw data files are not tracked in this Git repository to keep the repository lightweight.

## Dataset Details
* **Name:** SST-2 / SST-5 (Stanford Sentiment Treebank)
* **Source:** Hugging Face `datasets` library / Kaggle
* **Format:** Text strings with associated sentiment labels (binary or fine-grained)

## How to Access the Data
The data is automatically downloaded and cached locally via the Hugging Face `datasets` pipeline in the notebooks. If you need to manually download the source files, please download them from [Hugging Face Datasets](https://huggingface.co/datasets) or your specific source and place them in this directory as:
* `data/train.csv`
* `data/test.csv`