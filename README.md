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
