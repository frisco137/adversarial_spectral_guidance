
import os
import sys
import pickle
import numpy as np
import torch
import PIL.Image
import click
import tqdm
import subprocess
import shutil

# Add edm_base to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'edm_base'))

import dnnlib
from torch_utils import distributed as dist
from sampler_spectral import spectral_guidance_sampler

def generate_samples(net, outdir, num_samples, batch_size, num_steps, guidance_scale, profile_path, device):
    os.makedirs(outdir, exist_ok=True)
    
    seeds = range(num_samples)
    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_seeds_list = [seeds[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
    
    print(f"Generating {num_samples} samples to {outdir}...")
    
    for batch_seeds in tqdm.tqdm(batch_seeds_list, desc=f"Scale {guidance_scale} Steps {num_steps}"):
        bs = len(batch_seeds)
        latents = torch.randn([bs, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        
        class_labels = None
        if net.label_dim:
            class_labels = torch.zeros([bs, net.label_dim], device=device)
            class_labels[:, 0] = 1 # Dummy class 0
        
        # Run Sampling
        images = spectral_guidance_sampler(
            net, latents, class_labels, num_steps=num_steps,
            guidance_scale=guidance_scale, profile_path=profile_path
        )
        
        # Save images
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        
        for seed, img_np in zip(batch_seeds, images_np):
            if img_np.shape[2] == 1:
                PIL.Image.fromarray(img_np[:, :, 0], 'L').save(os.path.join(outdir, f'{seed:05d}.png'))
            else:
                PIL.Image.fromarray(img_np, 'RGB').save(os.path.join(outdir, f'{seed:05d}.png'))

def compute_fid(images_dir, ref_path):
    print(f"Computing FID for {images_dir}...")
    # Using python -m style or direct path
    fid_script = os.path.join('edm_base', 'fid.py')
    
    # Check if fid script exists
    if not os.path.exists(fid_script):
        print("Error: fid.py not found in edm_base.")
        return None
        
    cmd = [
        sys.executable, fid_script, 'calc',
        '--images', images_dir,
        '--ref', ref_path
    ]
    
    # We want to capture the output to parse the FID score
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Output format usually has "frechet_inception_distance: 12.34" or similar at the end
        print(result.stdout)
        
        # Simple parsing logic (might need adjustment based on edm output)
        # EDM fid.py outputs JSON-like metric dict usually? Or just prints.
        # Let's just return the full output for the user to see or regex it.
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"FID Calculation failed: {e}")
        print(e.stderr)
        return None

@click.command()
@click.option('--network', 'network_pkl', required=True, help='Path to model pickle')
@click.option('--profile', 'profile_path', default='spectral_profile.npy', help='Path to spectral profile')
@click.option('--ref', 'ref_path', default='models/imagenet-64x64.npz', help='Path to FID reference stats')
@click.option('--base_out', default='sweep_results', help='Base directory for results')
@click.option('--samples', type=int, default=1000, help='Samples per config')
@click.option('--batch', type=int, default=64, help='Batch size')
def main(network_pkl, profile_path, ref_path, base_out, samples, batch):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load network once
    print(f"Loading network: {network_pkl}")
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
        
    # Parameter Grid
    scales = [0.5, 1.5, 3.0, 5.0]
    steps_list = [20, 40]
    
    results = []
    
    for steps in steps_list:
        for scale in scales:
            config_name = f"scale_{scale}_steps_{steps}"
            outdir = os.path.join(base_out, config_name)
            
            print(f"\n=== Running Config: {config_name} ===")
            
            # 1. Generate
            if os.path.exists(outdir) and len(os.listdir(outdir)) >= samples:
                print(f"Directory {outdir} already has files. Skipping generation (or assuming done).")
            else:
                generate_samples(
                    net, outdir, samples, batch, steps, scale, profile_path, device
                )
            
            # 2. Compute FID
            if os.path.exists(ref_path):
                output = compute_fid(outdir, ref_path)
                results.append((config_name, output))
            else:
                print(f"Reference stats {ref_path} not found. Skipping FID.")
                results.append((config_name, "N/A"))
                
    # Summary
    print("\n\n=== SWEEP SUMMARY ===")
    with open(os.path.join(base_out, 'summary.txt'), 'w') as f:
        for name, output in results:
            summary_line = f"Config: {name}\nOutput:\n{output}\n{'-'*20}\n"
            print(summary_line)
            f.write(summary_line)
            
if __name__ == "__main__":
    main()
