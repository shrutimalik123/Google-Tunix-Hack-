import subprocess
import os

packages = [
    "clu>=0.0.12",
    "ml_collections",
    "absl-py",
    "chex"
]

print("--- Downloading Missing Packages One by One ---")
for pkg in packages:
    print(f"Downloading {pkg}...")
    cmd = (
        f"pip download "
        f"--dest packages "
        f"--platform manylinux2014_x86_64 "
        f"--python-version 3.10 "
        f"--implementation cp "
        f"--abi cp310 "
        f"--only-binary=:all: "
        f"{pkg}"
    )
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"SUCCESS: {pkg}")
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: {pkg} - trying no binaries restriction")
        # Fallback for pure python packages if no wheel matches
        cmd_fallback = (
            f"pip download "
            f"--dest packages "
            f"--no-deps "
            f"{pkg}"
        )
        try:
           subprocess.check_call(cmd_fallback, shell=True)
           print(f"SUCCESS (fallback): {pkg}")
        except subprocess.CalledProcessError as e2:
           print(f"FAILURE (fallback): {pkg}")

