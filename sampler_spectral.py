
import torch
import numpy as np
import os
from spectral_utils import create_dampening_mask, apply_spectral_perturbation

def spectral_guidance_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    guidance_scale=1.5, profile_path='spectral_profile.npy'
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Load Spectral Profile
    spectral_profile = None
    if os.path.exists(profile_path):
        try:
            spectral_profile = np.load(profile_path)
            # Ensure shape matches num_steps? 
            # If profile has different steps, we might need interpolation or error.
            # Assuming profile was generated with same num_steps.
            if spectral_profile.shape[0] != num_steps:
                print(f"Warning: Spectral profile steps ({spectral_profile.shape[0]}) != num_steps ({num_steps}). Guidance may be misaligned.")
        except Exception as e:
            print(f"Error loading spectral profile: {e}")
    else:
        print(f"Warning: Spectral profile '{profile_path}' not found. Running without guidance.")

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Standard Estimate D(x)
        # EDM output is D(x; sigma) which is the clean image prediction.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        
        # --- Spectral Guidance Block ---
        if spectral_profile is not None and guidance_scale != 0:
            # Get profile vector for this step
            # Handle index if profile is smaller/larger (clamp)
            prof_idx = min(i, len(spectral_profile) - 1)
            profile_vec = spectral_profile[prof_idx]
            
            # Create mask
            # img_shape (H, W)
            h, w = latents.shape[-2:]
            mask = create_dampening_mask(profile_vec, (h, w)).to(latents.device)
            
            # Create "Bad" version
            # We perturb the *prediction* (denoised), not x_hat.
            # Rationale: We want to guide away from the "active frequencies" in the current estimate.
            x_bad = apply_spectral_perturbation(denoised, mask)
            
            # Calculate guidance direction
            # We want to move AWAY from x_bad.
            # Standard CFG: D = D_uncond + s * (D_cond - D_uncond)
            # Here "Cond" is what we have (denoised), "Uncond" is "Bad"?
            # Or Negative Prompting logic: D_final = D + s * (D - D_negative)
            # Here x_bad is the negative target.
            
            delta = denoised - x_bad
            denoised_final = denoised + guidance_scale * delta
        else:
            denoised_final = denoised
            
        # Euler step using guided denoised
        d_cur = (x_hat - denoised_final) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction (Heun)
        # Note: We usually re-evaluate the network at the next step.
        # Should we apply guidance there too?
        # Yes, for consistency.
        
        if i < num_steps - 1:
            # 1st order guess
            x_prime = x_next # This is x_{i+1} from Euler
            
            # Evaluate at x_prime, t_next
            denoised_prime = net(x_prime, t_next, class_labels).to(torch.float64)
            
            # --- Spectral Guidance Block for 2nd order ---
            if spectral_profile is not None and guidance_scale != 0:
                # Use next step profile? Or same?
                # Heun averages slopes. t_next is the next time step.
                # So we should conceptually use profile[i+1]?
                # But the step index for the loop is i. 
                # Let's use profile[i+1] if available, else profile[i].
                prof_idx_next = min(i + 1, len(spectral_profile) - 1)
                profile_vec_next = spectral_profile[prof_idx_next]
                
                mask_next = create_dampening_mask(profile_vec_next, (h, w)).to(latents.device)
                x_bad_prime = apply_spectral_perturbation(denoised_prime, mask_next)
                
                delta_prime = denoised_prime - x_bad_prime
                denoised_prime_final = denoised_prime + guidance_scale * delta_prime
            else:
                denoised_prime_final = denoised_prime
            
            d_prime = (x_prime - denoised_prime_final) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
