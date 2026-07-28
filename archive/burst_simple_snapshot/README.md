# Compositional Capabilities of Autoregressive Transformers: A Study on Synthetic, Interpretable Tasks

Paper: [Compositional Capabilities of Autoregressive Transformers](https://openreview.net/forum?id=LhT1YHmyCN)

**Summary.** We create a synthetic setup to evaluate the ability of autoregressive Transformers to learn function compositions. We find that: (1) Autoregressive Transformers learn function compositions using very compositions in the training data (unlike LSTMs); (2) generating intermediate outputs when composing functions is more effective for generalizing to new, unseen compositions; (3) the attention layers select which function to apply while the feed-forward layers execute the selected capability. 

## Setup

### Option A: uv (recommended)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) then run:

```bash
uv sync
```

This creates a `.venv` with all pinned dependencies (including CUDA 11.8 PyTorch). Run scripts with:

```bash
uv run python 01_generate_data.py
```

Or activate the venv directly:

```bash
source .venv/bin/activate
python 01_generate_data.py
```

For notebooks in VS Code / Cursor, select the interpreter at `.venv/bin/python` or the **Python (composition-uv)** kernel.

### Option B: micromamba

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) then run:

```bash
micromamba create -y -f env.yml
micromamba activate composition
```

# Usage

**Step 1**: Generate training data using `01_generate_data.py`. The config file `config/gen/conf.yaml` can be modified to generate prompts in the direct or step-by-step formats. The config file also controls other choices like the number of in-order or out-of-order compositions. 

**Step 2**: Train model using `02_train.py`. Modify `config/train/conf.yaml` to use the data generated in step 1.

**Step 3**: Evaluate data on in-order (`03_evaluate_i.py`) or out-of-order (`03_evaluate_o.py`) compositions. Note that during evaluation, the model must autoregressively generate the outputs. Modify 


```bash
python 01_generate_data.py
python 02_train.py
python 03_evaluate_i.py
```

The default config runs all 3 steps in less than 10 minutes.

## Directory structure

```
├── 01_generate_data.py. # Generate train data
├── 02_train.py                   # Train networks
├── 03_evaluate_i.py         # Evaluating in-order functions
├── 03_evaluate_o.py.       # Evaluating out-of-order functions
├── env.yml                          # Environment files 
├── config/                           # Config files
├── net/                                 # Training scripts and architectures
│   ├── lstm.py
│   ├── nanogpt.py               
│   └── runner.py                   # Training scripts for Transformer
├── run.sh
└── synthetic
    ├── functions.py.    # Create functions and compositions
    ├── generator.py.    # Generate prompts for training and eval
    └── init.py                # Load config and set random seed
```
