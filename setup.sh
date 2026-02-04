#!/bin/bash

# 1. Clone EDM Base
if [ ! -d "edm_base" ]; then
    echo "Cloning NVlabs/edm..."
    git clone https://github.com/NVlabs/edm.git edm_base
else
    echo "edm_base already exists."
fi

# 2. Create models directory
mkdir -p models

# 3. Download Model Checkpoints
# Using the Conditional ADM ImageNet-64 model as the unconditional VP one was 403 Forbidden.
# Also downloading reference statistics for FID.

echo "Downloading models..."
wget -nc -P models https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
wget -nc -P models https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz

echo "Setup complete. Directory structure:"
ls -F
