# Tunix Reasoning Trainer: Fully Offline & Fixed

**Use this notebook to train Gemma 2 on TPUs without internet access.**

## Overview
This project provides a robust, fully functional, offline-capable training pipeline for the Gemma 2 model using Google's `Tunix` library. It was engineered specifically to overcome the strict "No Internet" constraints of secure Kaggle environments.

## Key Challenges Solved
Running Tunix offline presented several critical blockers that this notebook resolves:

1.  **Read-Only Filesystem Errors**
    *   *Problem:* `pip install` fails on Kaggle's input directories because it tries to write build artifacts.
    *   *Solution:* Implemented a structured workaround that mirrors the Tunix source code to the writable `/tmp` directory before installation, allowing the build process to complete successfully.

2.  **Dependency Hell & Missing Wheels**
    *   *Problem:* The `google-metrax` library has hidden dependencies (`clu`, `ml_collections`) that generate runtime import errors if not present.
    *   *Solution:* I created a custom asset downloader script to pre-fetch these specific binary wheels, ensuring a complete dependency graph is available offline.

3.  **Broken Imports & APIs**
    *   *Problem:* The provided example documentation referenced `tunix.config` and `qwix.LoraConfig`, which do not exist in the current library version, leading to immediate `ImportError` and `AttributeError` crashes.
    *   *Solution:* I reverse-engineered the API by analyzing the source code (`tests/test_common.py`), identifying the correct `PeftTrainer` and `qwix.LoraProvider` interfaces. I essentially rewrote the training initialization logic to match the actual library implementation.

4.  **Missing Data Loader**
    *   *Problem:* The example `text_dataset` module was not included in the pip package.
    *   *Solution:* I implemented a drop-in `TextDataset` class directly within the notebook to handle `.jsonl` streaming and SentencePiece tokenization, ensuring data flows correctly to the model.

## How to Use
1.  **Import Dataset:** Add the `tunix-offline-assets` dataset to your kernel.
2.  **Disable Internet:** Ensure the "Internet" toggle is set to **Off**.
3.  **Run:** Execute the notebook. It will automatically detect the offline environment, install dependencies from the local dataset, and begin fine-tuning Gemma 2 on the TPU.

## Components
*   **`Tunix_Reasoning_Trainer.ipynb`**: The main notebook for training.
*   **`download_assets.py`**: A utility script to download all necessary wheels and the model for offline use.
*   **`data_prep.py`**: Prepares the GSM8K dataset in the required format.

## Dependencies (Offline)
All dependencies are provided in the `tunix-offline-assets` dataset, including:
*   `tunix` (cloned source)
*   `flax`, `jax`, `optax`
*   `google-metrax`, `clu`, `ml_collections`
*   `sentencepiece`
