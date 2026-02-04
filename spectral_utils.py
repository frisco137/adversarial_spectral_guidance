import torch
import torch.fft
import numpy as np

def get_rapsd(image_tensor):
    """
    Computes the Radially Averaged Power Spectral Density (RAPSD) of a batch of images.
    
    Args:
        image_tensor (torch.Tensor): Tensor of shape (B, C, H, W).
        
    Returns:
        torch.Tensor: Profile of shape (B, C, N//2) where N is min(H, W).
    """
    b, c, h, w = image_tensor.shape
    n = min(h, w)
    
    # Compute FFT
    # fft2 returns standard fft, DC at top-left
    fft_img = torch.fft.fft2(image_tensor, dim=(-2, -1))
    
    # Shift DC to center
    fft_img = torch.fft.fftshift(fft_img, dim=(-2, -1))
    
    # Power spectrum
    power_spectrum = torch.abs(fft_img) ** 2
    
    # Create coordinate grid relative to center
    y_idx = torch.arange(h, device=image_tensor.device) - h // 2
    x_idx = torch.arange(w, device=image_tensor.device) - w // 2
    y, x = torch.meshgrid(y_idx, x_idx, indexing='ij')
    radius = torch.sqrt(x**2 + y**2).long() # integer radius
    
    # Radial averaging
    # We want to average power at each integer radius r from 0 to n//2
    # This involves a bit of scatter_add or loop. 
    # Since n is small (32 for 64x64), a loop or vectorization is fine.
    # Vectorized approach:
    
    max_r = n // 2
    
    # Flatten spatial dims to facilitate grouping
    power_flat = power_spectrum.reshape(b, c, -1) # (B, C, H*W)
    radius_flat = radius.reshape(-1) # (H*W)
    
    # Mask for valid radii
    mask = radius_flat < max_r
    
    valid_radii = radius_flat[mask]
    valid_power = power_flat[:, :, mask]
    
    # We can use scatter_add if we pre-allocate
    # OR, since we want average, we need sum and count.
    
    profile = torch.zeros(b, c, max_r, device=image_tensor.device)
    count = torch.zeros(max_r, device=image_tensor.device)
    
    # Add count (ones)
    ones = torch.ones_like(valid_radii, dtype=torch.float32)
    count.scatter_add_(0, valid_radii, ones)
    
    # Add power
    # We need to broadcast scatter_add across batch/channel
    # Actually, let's just loop over radii for simplicity and readability unless slow.
    # 64x64 is small. max_r = 32. A loop of 32 is negligible compared to diffusion steps.
    
    # Alternative: bincount for specific 1D but we have B,C.
    # Let's stick to a robust standard implementation.
    
    # Pre-compute masks for each r to avoid loop in sampler if possible? No, input changes.
    # But h,w is constant.
    
    # Optimization: One-time pre-computation of indices for each r would be faster,
    # but for now let's implement the logic clearly.
    
    # Let's use the loop for clarity and safety with B,C dims
    
    profiles = []
    for r in range(max_r):
        # Create mask for this radius
        mask_r = (radius == r)
        # Mean over spatial dimensions where mask is true
        # power_spectrum: (B, C, H, W)
        # mask_r: (H, W)
        
        # We want to select elements and mean them.
        # masked_select returns 1D flattened. We need to preserve B, C.
        
        # Taking value at mask:
        # We can sum(power * mask) / sum(mask)
        
        val_sum = (power_spectrum * mask_r.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))
        val_count = mask_r.sum()
        
        # Avoid div by zero
        val_mean = val_sum / (val_count + 1e-8)
        profiles.append(val_mean)
        
    # Stack: (B, C, max_r)
    return torch.stack(profiles, dim=-1)

def create_dampening_mask(profile_vector, img_shape):
    """
    Creates a 2D Fourier mask based on the spectral profile.
    
    Args:
        profile_vector (torch.Tensor or np.ndarray): 1D array of length N//2 describing active freqs.
        img_shape (tuple): (H, W).
        
    Returns:
        torch.Tensor: 2D mask (H, W) values in [0, 1].
    """
    h, w = img_shape
    if isinstance(profile_vector, np.ndarray):
        profile_vector = torch.from_numpy(profile_vector)
        
    # Normalize profile to [0, 1]
    # We assume higher value in profile = more active = needs more dampening.
    # So mask should be low (close to 0) where profile is high.
    
    # Working in linear or log space?
    # User note: "work in Log Space before normalizing"
    
    # Avoid log(0)
    p_log = torch.log10(profile_vector + 1e-8)
    
    # Normalize to [0, 1] relative to the vector's range
    p_min = p_log.min()
    p_max = p_log.max()
    
    if p_max - p_min < 1e-8:
        p_norm = torch.zeros_like(p_log)
    else:
        p_norm = (p_log - p_min) / (p_max - p_min)
        
    # Invert to create dampening factor (1 = keep, 0 = remove)
    v_damp = 1.0 - p_norm
    
    # Ensure it is on the right device if needed, but we usually return CPU or same device
    # Let's assume we construct the mask on the same device as profile_vector
    
    # Create 2D grid of radii
    y_idx = torch.arange(h, device=profile_vector.device) - h // 2
    x_idx = torch.arange(w, device=profile_vector.device) - w // 2
    y, x = torch.meshgrid(y_idx, x_idx, indexing='ij')
    radius = torch.sqrt(x**2 + y**2).long()
    
    # Map radius to value
    # Clamp radius to valid range of profile_vector indices
    max_idx = len(profile_vector) - 1
    radius_clamped = radius.clamp(max=max_idx)
    
    # Gather values
    mask = v_damp[radius_clamped]
    
    # Handle corners/edges where radius > max_idx?
    # Usually we want to dampen high freqs if we don't know?
    # Or keep them?
    # EDM 64x64: max radius to corner is ~45. N//2 is 32.
    # The profile is only defined up to 32.
    # Let's assume the outer corners get the value of the last bin (high freq).
    
    return mask

def apply_spectral_perturbation(x, mask):
    """
    Applies the spectral dampening mask to the latent x.
    
    Args:
        x (torch.Tensor): Latent image (B, C, H, W).
        mask (torch.Tensor): Mask (H, W) or (B, C, H, W).
        
    Returns:
        torch.Tensor: Perturbed image.
    """
    # FFT
    z = torch.fft.fft2(x, dim=(-2, -1))
    z_shifted = torch.fft.fftshift(z, dim=(-2, -1))
    
    # Apply mask
    # Mask should broadcast. If mask is (H, W), it broadcasts to (B, C, H, W).
    z_masked = z_shifted * mask
    
    # IFFT
    z_unshifted = torch.fft.ifftshift(z_masked, dim=(-2, -1))
    x_out = torch.fft.ifft2(z_unshifted, dim=(-2, -1))
    
    # Return Real part
    return x_out.real
