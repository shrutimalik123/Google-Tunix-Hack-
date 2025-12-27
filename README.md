# 🏆 Tunix Reasoning Trainer: Fully Offline & Fixed

**First Prize Submission for the Google Tunix Hackathon**

This project demonstrates how to **train a Reasoning Model (Gemma 2) completely offline** on Kaggle TPUs. It is engineered to solve the specific "dependency hell" that prevents modern JAX/Flax libraries from running in secure, internet-disabled environments.

![Thumbnail](./tunix_submission_thumbnail.png)

## 📺 Demo
[**Watch the 3-Minute Walkthrough on YouTube**](YOUR_YOUTUBE_LINK_HERE)

---

## 🧠 The "Why": Offline Engineering Logic
Most participants will use the default Kaggle environment, which limits them to pre-installed libraries or requires internet access. However, real-world secure AI training (like in healthcare or finance) often happens in air-gapped systems. 

**This project creates a reproducible "Air-Gapped" training pipeline.**

### 1. The Challenge: File System & Build Artifacts
*   **Problem:** Python's `pip install` usually downloads files to a temporary cache and builds "wheels". On Kaggle, the input directories (`/kaggle/input`) are **read-only**.
*   **The Fail State:** If you try to `pip install ./local_package`, it fails because it cannot write `egg-info` or build files to the source directory.
*   **Our Solution (Source Injection):**
    We perform a **runtime injection** in the notebook.
    ```python
    # Copy read-only source to writable /tmp
    !cp -r /kaggle/input/tunix-offline-assets/packages/tunix /tmp/tunix
    # Install from the writable location
    !pip install /tmp/tunix/.[tpu] --no-index ...
    ```
    This "tricks" pip into treating the package as a local editable install in the ephemeral `/tmp` storage, bypassing the read-only restriction.

### 2. The Challenge: Hidden Dependencies
*   **Problem:** The `tunix` library depends on Google-internal tools like `google-metrax`, `clu`, and `ml_collections`. These are NOT in the standard Kaggle image and pip cannot fetch them offline.
*   **Our Solution (Binary Mirroring):**
    We wrote a custom `download_assets.py` script that acts as a **local PyPI mirror**. It:
    *   Reads `requirements.txt`.
    *   Forces the download of `manylinux2014_x86_64` binary wheels (compatible with Kaggle's Linux environment).
    *   These are stored in `packages/` and uploaded as a dataset, guaranteeing zero runtime downloads.

### 3. The Challenge: Broken Data Loaders
*   **Problem:** The `tunix` library's `text_dataset` module was missing or incompatible in the distributed source.
*   **Our Solution (JAX Native Loader):**
    Instead of relying on a black-box loader, we implemented a custom `TextDataset` class directly in the notebook.
    *   It streams `.jsonl` data line-by-line (memory efficient).
    *   It uses `sentencepiece` directly to tokenize on the fly.
    *   It yields JAX-ready `int32` arrays, removing the need for complex PyTorch/TensorFlow bridges.

---

## 📈 Results: "Reasoning" Capabilities
We fine-tuned **Gemma 2 2B Instruct** on the GSM8K (Grade School Math) dataset.
*   **Technique:** LoRA (Low-Rank Adaptation) via `qwix`.
*   **Steps:** 3000 (approx. 1 epoch).
*   **Inference Loop:** We implemented a custom sampling loop that forces the model to output a `<reasoning>` trace before its final answer.

**Example Output:**
> **User:** If I have 5 apples and eat 2, how many are left?
> **Model:** <reasoning> Start with 5. Subtract 2. 5 - 2 = 3. </reasoning> <answer> 3 </answer>

---

## 📂 Reproduction Steps

1.  **Clone & Prep:**
    ```bash
    git clone https://github.com/shrutimalik123/Google-Tunix-Hack-.git
    cd Google-Tunix-Hack-
    # Authenticates with Hugging Face to get Gemma 2
    python download_assets.py
    ```

2.  **Kaggle Setup:**
    *   Create a **Dataset** named `tunix-offline-assets` containing the `packages/` folder.
    *   Create a **Model** named `gemma-2-2b-hackathon` containing the `model/gemma-2-2b-it` folder.

3.  **Run Training:**
    *   Upload `Tunix_Reasoning_Trainer.ipynb`.
    *   Add your Dataset and Model.
    *   **Turn Internet OFF.**
    *   Run All.

---
*Built for the Google Tunix Hackathon by Shruti Malik.*
