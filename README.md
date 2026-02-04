# Spectral Guidance for Diffusion Models

This repository contains an implementation of "Spectral Guidance" for diffusion models on ImageNet-64.

## Setup

To set up the environment and download the necessary models, run:

```bash
./setup.sh
```

This will:
1. Clone the `NVlabs/edm` repository to `./edm_base`.
2. Download the ImageNet-64 model checkpoint and reference statistics to `./models`.

## Usage

### 1. Profile Spectrum
First, generate a spectral profile from the base model:

```bash
python profile_spectrum.py --network models/edm-imagenet-64x64-cond-adm.pkl --batch 32 --steps 40
```

### 2. Run Experiments

**Debug Mode (Visualization):**
```bash
python run_experiment.py --mode debug --network models/edm-imagenet-64x64-cond-adm.pkl
```

**Fidelity Mode (Evaluation):**
```bash
python run_experiment.py --mode fidelity --network models/edm-imagenet-64x64-cond-adm.pkl --total_samples 10000 --batch 64
```
