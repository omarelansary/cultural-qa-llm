# cultural-qa-llm
This project explores LLM-based approaches to cultural question answering using Meta’s Llama-3-8B model. The goal is to build and evaluate systems that can correctly interpret cultural and linguistic cues across multiple cultural contexts.  
We address both multiple-choice (MCQ) and short-answer (SAQ) tasks as defined in the Codabench benchmark, investigating techniques such as prompt engineering, parameter-efficient fine-tuning, self-consistency, and retrieval-augmented reasoning—while strictly adhering to the constraint of using Llama-3-8B as the sole language model.  
Model performance is evaluated using benchmark accuracy, with particular attention to generalization across cultures and the limitations of exact-match evaluation for generative answers.

## Project Pipeline

![Project Pipeline](media/images/pipeline-diagram.png)

## Directory Structure & Guide

### 1. Where things live (The "Do Not Touch" Rules)
* **`src/`**: **Main Codebase.** All reusable Python logic goes here.
    * `data_loader.py`: Only logic for reading/parsing TSVs.
    * `model.py`: Llama-3 loading logic (quantization configs go here).
    * `utils.py`: formatting logic for the specific Codabench TSV requirements.
* **`configs/`**: **Hyperparameters.** ⚠️ **Do not hardcode variables in Python files.** Use YAML files here (e.g., `mcq_baseline.yaml`) to set learning rates, batch sizes, etc.
* **`prompts/`**: **Prompt Templates.** Store prompt text strings here (e.g., `zero_shot_prompt.txt`) so we can version control our prompt engineering experiments.

### 2. Local Only (Gitignored)
* **`data/`**: Place the raw OPAL datasets here.
* **`output/`**: All heavy outputs (model checkpoints, logs, vector DBs) go here. **Never push this folder.**

### 3. Execution & Submission
* **`scripts/`**: Contains `.slurm` scripts. Use these to submit jobs to the HPC queue.
* **`submission/`**: Run the scripts here to generate the final `.zip` file for Codabench.



## Quick Start (HPC)

1. **Initialize Environment:**
   ```bash
   make setup  # OR bash scripts/setup_env.sh
   ```

2. **Run a Baseline Experiment:**
   ```
   bash
   sbatch scripts/train_mcq.slurm
   ```

## Dependency Management (HPC-friendly)

- **Add a new library** (updates `pyproject.toml` + `requirements.txt`, optional install):
  ```bash
  python scripts/manage_deps.py add "package==1.2.3" --pip-install
  ```
- **Remove a library** (updates both files, optional uninstall):
  ```bash
  python scripts/manage_deps.py remove package-name --pip-uninstall
  ```
- **Freeze requirements** (regenerate `requirements.txt` from `pyproject.toml`):
  ```bash
  python scripts/manage_deps.py freeze
  ```
  Use this after any manual edit to `pyproject.toml` or before pushing so CI/HPC installs match.
- **Setup env on HPC** (loads modules, freezes, installs editable):
  ```bash
  bash scripts/setup_env.sh
  ```
  This runs `manage_deps.py freeze` first to keep `requirements.txt` in sync, then `pip install -e .` and installs the generated requirements inside your conda env.
