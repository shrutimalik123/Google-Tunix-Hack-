from huggingface_hub import hf_hub_download
import os
import shutil

repo_id = "google/gemma-2-2b-it"
filename = "tokenizer.model"
local_dir = os.path.abspath("model/gemma-2-2b-it")

print(f"Attempting to download {filename} from {repo_id}...")

try:
    file_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        cache_dir=os.path.join(local_dir, ".cache"),
        token=True
    )
    print(f"Downloaded to cache: {file_path}")
    
    # Manually copy to target
    target_path = os.path.join(local_dir, filename)
    shutil.copy(file_path, target_path)
    print(f"Copied to: {target_path}")
    print("Success!")

except Exception as e:
    print(f"Download failed: {e}")
