
import os
import sys
import pickle
import numpy as np
import torch
import torch.fft
import PIL.Image
import click
import tqdm

# Add edm_base to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'edm_base')))

import dnnlib
from torch_utils import distributed as dist
from spectral_utils import create_dampening_mask, apply_spectral_perturbation
from sampler_spectral import spectral_guidance_sampler

def save_image_grid(img, fname, drange=[0,1], grid_size=None):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def run_debug(net, outdir, profile_path, num_steps=10, batch_size=4, device='cuda'):
    print("Running Debug Mode...")
    os.makedirs(outdir, exist_ok=True)
    
    # Load profile
    if not os.path.exists(profile_path):
        print(f"Profile {profile_path} not found. Cannot debug spectral guidance.")
        return
    spectral_profile = np.load(profile_path)
    
    # Setup sampler params
    sigma_min = 0.002
    sigma_max = 80
    rho = 7
    S_churn = 0
    S_min = 0
    S_max = float('inf')
    S_noise = 1
    guidance_scale = 5.0 # High scale for visualization
    
    # Adjust noise levels
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time steps
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Latents
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    x_next = latents.to(torch.float64) * t_steps[0]
    
    # Labels
    class_labels = None
    if net.label_dim:
        class_labels = torch.zeros([batch_size, net.label_dim], device=device)
        class_labels[:, 0] = 1
        
    # We will capture visualizations for a few steps
    capture_steps = [0, num_steps // 2, num_steps - 2]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        
        # Guidance Logic
        prof_idx = min(i, len(spectral_profile) - 1)
        profile_vec = spectral_profile[prof_idx]
            
        h, w = latents.shape[-2:]
        mask = create_dampening_mask(profile_vec, (h, w)).to(device)
            
        x_bad = apply_spectral_perturbation(denoised, mask)
        delta = denoised - x_bad
        denoised_guided = denoised + guidance_scale * delta
        
        # Save visualization
        if i in capture_steps:
            print(f"Capturing step {i}...")
            # Row: Original, Perturbed, Guided
            # Stack images
            vis_images = torch.cat([denoised, x_bad, denoised_guided], dim=0)
            # Normalize to [0,1] for save_image_grid (actually it expects raw floats and handles range)
            # EDM images are approx [-1, 1].
            save_image_grid(vis_images.cpu().numpy(), os.path.join(outdir, f'debug_step_{i}.png'), drange=[-1, 1], grid_size=(3, batch_size))
            
        # Continue step
        d_cur = (x_hat - denoised_guided) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # Correction (skipped for debug simplicity or keeping it?) 
        # Let's Skip 2nd order for debug visualization clarity unless crucial.
        # EDM usually needs Heun? OK let's keep it simple Euler for debug logic 
        # or we have to duplicate the whole Heun logic.
        # Prompt: "Verify that 'Perturbed Estimate' looks like spectrally deficient..."
        # Euler is enough to see the perturbation.
        
    print("Debug run complete.")


@click.command()
@click.option('--mode', type=click.Choice(['debug', 'fidelity']), required=True)
@click.option('--network', 'network_pkl', required=True)
@click.option('--outdir', type=str, default='experiment_out')
@click.option('--profile', 'profile_path', type=str, default='spectral_profile.npy')
@click.option('--batch', 'batch_size', type=int, default=64)
@click.option('--steps', 'num_steps', type=int, default=40)
@click.option('--total_samples', type=int, default=10000)
@click.option('--guidance_scale', type=float, default=1.5)
def main(mode, network_pkl, outdir, profile_path, batch_size, num_steps, total_samples, guidance_scale):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"Loading network: {network_pkl}")
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
        
    if mode == 'debug':
        run_debug(net, outdir, profile_path, num_steps=10, batch_size=4, device=device)
    
    elif mode == 'fidelity':
        print(f"Running Fidelity Mode: {total_samples} samples")
        os.makedirs(outdir, exist_ok=True)
        
        # Seeds
        seeds = range(total_samples)
        
        # Loop over batches
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        batch_seeds_list = [seeds[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
        
        for batch_seeds in tqdm.tqdm(batch_seeds_list):
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
        
        print("Done generating images. To calculate FID, run:")
        print(f"torchrun --standalone --nproc_per_node=1 edm_base/fid.py calc --images={outdir} --ref=models/imagenet-64x64.npz")

if __name__ == "__main__":
    main()
