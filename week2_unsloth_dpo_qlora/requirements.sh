#!/bin/bash
%%capture
# -------------------------------
# Step 1: Remove old PyTorch versions
# -------------------------------
pip3-autoremove torch torchvision torchaudio -y

# -------------------------------
# Step 2: Install specific versions to avoid recompilation
# -------------------------------
pip install --no-cache-dir torch==2.4.0 torchvision==0.15.2 torchaudio==2.4.0 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# -------------------------------
# Step 3: Install xformers and triton
# -------------------------------
pip install --no-cache-dir xformers==0.0.27.post2 triton

# -------------------------------
# Step 4: Install unsloth from GitHub
# -------------------------------
pip install --no-cache-dir "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"

# -------------------------------
# Step 5: Install additional dependencies
# -------------------------------
pip install --no-cache-dir datasets trl wandb

