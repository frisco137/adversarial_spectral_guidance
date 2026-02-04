
import os
import sys
import pickle
import numpy as np
import torch
import tqdm
import click

# Add edm_base to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'edm_base')))

import dnnlib
from torch_utils import distributed as dist
from spectral_utils import get_rapsd

# Re-implementing a basic EDM sampler hook for profiling
def profiling_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=40, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Store RAPSDs
    # shape: [num_steps, N//2]
    # We will accumulate average RAPSD over the batch
    rapsd_history = []

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        
        # --- PROFILING HOOK ---
        # Compute RAPSD of the *denoised* estimate (clean prediction)
        # denoised is (B, C, H, W)
        with torch.no_grad():
            # get_rapsd returns (B, C, R)
            # We average over Batch and Channel to get (R,) or keep Channel?
            # Prompt says "profile_vector: 1D array". So we average over B and C.
            # Using get_rapsd from spectral_utils
            current_rapsd = get_rapsd(denoised) # (B, C, R)
            current_rapsd_avg = current_rapsd.mean(dim=(0, 1)) # (R,)
            rapsd_history.append(current_rapsd_avg.cpu().numpy())
        # ----------------------

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return rapsd_history

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT', type=int, default=40, show_default=True)
@click.option('--batch', 'batch_size', help='Batch size', metavar='INT', type=int, default=64, show_default=True)
@click.option('--out', 'out_path', help='Output .npy file', metavar='FILE', type=str, default='spectral_profile.npy', show_default=True)
@click.option('--device', 'device', help='Device', metavar='STR', type=str, default='cuda')
def main(network_pkl, num_steps, batch_size, out_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using cpu")
        device = 'cpu'
    
    device = torch.device(device)
    
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    
    print(f'Generating profile with {num_steps} steps, batch size {batch_size}...')
    
    # Generate latents
    seeds = range(batch_size)
    # Using simple random latents
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    
    # Labels
    class_labels = None
    if net.label_dim:
        print(f"Model has {net.label_dim} classes. Using class 0.")
        class_labels = torch.zeros([batch_size, net.label_dim], device=device)
        class_labels[:, 0] = 1 # Use class 0
        
    # Run Sampler
    rapsd_history = profiling_sampler(net, latents, class_labels, num_steps=num_steps)
    
    # Process and Save
    # rapsd_history is list of (R,) numpy arrays.
    # Stack to (num_steps, R)
    profile_matrix = np.stack(rapsd_history)
    
    # Compute Delta if needed?
    # Prompt: "Calculate the Delta: RAPSD_t - RAPSD_{t+1}" OR "Alternative: Just store the raw RAPSD at t"
    # Prompt also says: "Save as spectral_profile.npy (Shape: [num_steps, radius])"
    # If we compute delta, we lose one step or pad?
    # Let's simple Save the RAW profile for now, and handle Delta in logic or just use Raw.
    # The prompt actually allowed "Just store the raw RAPSD".
    # And in sampler logic I implemented: "Identify Active Frequencies: Look up spectral_profile[step_index]".
    # If I store raw RAPSD, I am identifying active freqs as the ONES PRESENT.
    # This matches the alternative.
    
    np.save(out_path, profile_matrix)
    print(f'Saved spectral profile to {out_path} (Shape: {profile_matrix.shape})')

if __name__ == "__main__":
    main()
