import os
import subprocess
import shutil
import sys

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")

def download_assets():
    # 1. Prepare directories
    if not os.path.exists('packages'):
        os.makedirs('packages')
    if not os.path.exists('model'):
        os.makedirs('model')

    print("--- Processing Requirements ---")
    with open('requirements.txt', 'r') as f:
        lines = f.readlines()
    
    tunix_req = None
    other_reqs = []
    for line in lines:
        if 'tunix' in line.lower():
            tunix_req = line.strip() # keeps the git url
        else:
            other_reqs.append(line)
            
    with open('requirements_wheels.txt', 'w') as f:
        f.writelines(other_reqs)

    print("--- Downloading Binary Wheels (Linux x86_64) ---")
    pip_cmd_wheels = (
        "pip download "
        "--dest packages "
        "--platform manylinux2014_x86_64 "
        "--python-version 3.10 "
        "--implementation cp "
        "--abi cp310 "
        "--only-binary=:all: "
        "--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html "
        "-r requirements_wheels.txt"
    )
    run_command(pip_cmd_wheels)

    print("--- Downloading Tunix Source ---")
    # Manually clone Tunix since pip download git+... behavior can be flaky with destinations on Windows
    tunix_dest = os.path.join('packages', 'tunix')
    if os.path.exists(tunix_dest):
        print(f"Tunix source already exists at {tunix_dest}, updating...")
        run_command(f"git -C {tunix_dest} pull")
    else:
        print("Cloning Tunix...")
        run_command(f"git clone https://github.com/google/tunix.git {tunix_dest}")

    print("\n--- Downloading Gemma Model ---")
    try:
        from huggingface_hub import snapshot_download
        model_id = "google/gemma-2-2b-it"
        local_dir = "model/gemma-2-2b-it"
        if os.path.exists(os.path.join(local_dir, "config.json")):
             print("Model appears to be present.")
        else:
            print(f"Downloading {model_id} to {local_dir}...")
            snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
            print("Model download complete.")
    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    download_assets()
