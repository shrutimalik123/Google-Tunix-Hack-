import json

notebook_path = "Tunix_Reasoning_Trainer.ipynb"

# 1. Load Notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# --- Helper to create cells ---
def create_markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def create_code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

# --- 2. Offline Architecture Markdown ---
offline_arch_md = [
    "## 💡 Deep Dive: The Offline Training Architecture\n",
    "\n",
    "Running advanced training frameworks like Tunix in a strictly offline environment requires significant engineering. This notebook implements a unique \"source-injection\" pattern to bypass Kaggle's restrictions.\n",
    "\n",
    "### The Challenge\n",
    "Standard `pip install` commands fail in offline Kaggle input directories because they attempt to write build artifacts (egg-info, wheels) to the read-only file system. Since we cannot access PyPI, we must rely on pre-downloaded wheels and source code.\n",
    "\n",
    "### The Solution Phase\n",
    "1.  **Wheel Injection**: We utilize a custom `tunix-offline-assets` dataset containing `manylinux2014_x86_64` wheels for all hidden dependencies (including `google-metrax`, `clu`, `ml_collections`, `chex`).\n",
    "2.  **Source Mirroring**: The `tunix` source code is cloned into the input dataset. At runtime, we execute a critical step:\n",
    "    ```bash\n",
    "    cp -r /kaggle/input/tunix-offline-assets/packages/tunix /tmp/tunix\n",
    "    ```\n",
    "    This mirrors the library to a writable directory (`/tmp`), allowing `pip install` to build the package metadata locally without triggering file system errors.\n",
    "3.  **Custom Data Loading**: To avoid dependency on potentially version-mismatched bundled loaders, we implement a stripped-down, high-performance JAX data loader (`TextDataset`) directly in the notebook. This ensures our data pipeline is transparent and 100% compatible with the offline `sentencepiece` tokenizer.\n",
    "\n",
    "This architecture ensures that **Gemma 2 training is reproducible, stable, and completely self-contained.**"
]

# Insert after "intro" cell (index 1)
cells.insert(1, create_markdown(offline_arch_md))

# --- 3. Scale Training ---
# Find cell with "TrainingConfig(" and replace max_steps
found_config = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "TrainingConfig(" in source and "max_steps=50" in source:
            cell['source'] = [line.replace("max_steps=50,", "max_steps=3000,") for line in cell['source']]
            cell['source'] = [line.replace("eval_every_n_steps=10,", "eval_every_n_steps=100,") for line in cell['source']]
            found_config = True
if not found_config:
    print("Warning: TrainingConfig cell not found or matched.")

# --- 4. Add trainer.close() ---
found_train = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "trainer.train(" in source:
             # Append trainer.close() at the end
             cell['source'].append("\n    # Ensure checkpoints are final and resources released\n")
             cell['source'].append("    print('Closing trainer and saving final state...')\n")
             cell['source'].append("    trainer.close()\n")
             found_train = True
if not found_train:
    print("Warning: Training loop cell not found.")

# --- 5. Implement Inference ---
inference_code = [
    "# --- 6. Inference with Tunix Sampler ---\n",
    "from tunix.generate import sampler\n",
    "\n",
    "print(\"Initializing Inference Sampler...\")\n",
    "\n",
    "# 1. Define Cache Config\n",
    "# Ensure compatibility with the Gemma 2B model architecture\n",
    "cache_config = sampler.CacheConfig(\n",
    "    cache_size=2048,  # Max context window for generation\n",
    "    num_layers=model_config.num_layers,\n",
    "    num_kv_heads=model_config.num_kv_heads,\n",
    "    head_dim=model_config.head_dim\n",
    ")\n",
    "\n",
    "# 2. Initialize Sampler\n",
    "# We reuse the 'dataset.tokenizer' which is our loaded sentencepiece processor\n",
    "inference_sampler = sampler.Sampler(\n",
    "    transformer=lora_model,\n",
    "    tokenizer=dataset.tokenizer,\n",
    "    cache_config=cache_config\n",
    ")\n",
    "\n",
    "# 3. Define Test Prompts (Math Reasoning)\n",
    "prompts = [\n",
    "    \"User: If I bake 12 cookies and eat 3, then bake 6 more, how many do I have?\\nModel:\",\n",
    "    \"User: Calculate 25 * 4 + 10.\\nModel:\"\n",
    "]\n",
    "\n",
    "# 4. Generate\n",
    "print(f\"Generating responses for {len(prompts)} prompts...\")\n",
    "output = inference_sampler(\n",
    "    input_strings=prompts,\n",
    "    max_generation_steps=256,\n",
    "    temperature=0.7,\n",
    "    top_k=50,\n",
    "    echo=True,  # Print the prompt + completion\n",
    "    return_logits=False\n",
    ")\n",
    "\n",
    "# 5. Display Results\n",
    "for i, text in enumerate(output.text):\n",
    "    print(\"=\" * 40)\n",
    "    print(f\"PROMPT {i+1} RESULT:\")\n",
    "    print(\"=\" * 40)\n",
    "    print(text)\n",
    "    print(\"\\n\")"
]

found_inference = False
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Inference would go here" in source:
            cells[i] = create_code([line + "\n" for line in inference_code])
            found_inference = True
            break
if not found_inference:
    # If not found, append to end
    cells.append(create_code([line + "\n" for line in inference_code]))

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook upgraded successfully.")
