
import torch
import numpy as np
from spectral_utils import get_rapsd, create_dampening_mask, apply_spectral_perturbation

def test_rapsd():
    print("Testing get_rapsd...")
    # Create a simple image with known freq: e.g. a sine wave
    B, C, H, W = 1, 1, 64, 64
    x = torch.zeros(B, C, H, W)
    
    # Add a low frequency component
    # Center is 32, 32.
    # Sine wave along x axis
    # sin(2*pi * f * x / W)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # Frequency f=4
    x = torch.sin(2 * np.pi * 4 * xx / W)
    x = x.unsqueeze(0).unsqueeze(0)
    
    rapsd = get_rapsd(x)
    print(f"RAPSD shape: {rapsd.shape}")
    
    # Peak should be around r=4?
    # RAPSD averages over radius.
    # Freq 4 in spatial domain corresponds to peak at radius 4 in freq domain?
    # FFT indexing: k goes 0 to N/2.
    # sin(2pi k x / N) -> peak at index k.
    # So we expect peak at index 4.
    
    profile = rapsd[0, 0]
    peak_idx = torch.argmax(profile).item()
    print(f"Peak index: {peak_idx}")
    assert peak_idx == 4, f"Expected peak at 4, got {peak_idx}"
    print("RAPSD Test Passed.")

def test_mask():
    print("Testing create_dampening_mask...")
    profile = torch.zeros(32)
    profile[4] = 100.0 # High value at r=4
    
    mask = create_dampening_mask(profile, (64, 64))
    print(f"Mask shape: {mask.shape}")
    
    # Mask should be low at r=4 (due to inversion)
    # Check a point at radius 4
    # Center (32, 32). (32, 36) has r=4.
    mid = 32
    val_at_r4 = mask[mid, mid+4].item()
    val_at_r10 = mask[mid, mid+10].item()
    
    print(f"Mask at r=4 (active): {val_at_r4}")
    print(f"Mask at r=10 (inactive): {val_at_r10}")
    
    assert val_at_r4 < val_at_r10, "Mask should dampen active frequencies more."
    print("Mask Test Passed.")

def test_perturbation():
    print("Testing apply_spectral_perturbation...")
    x = torch.randn(1, 1, 64, 64)
    mask = torch.ones(64, 64) * 0.5 # Uniform dampening
    
    out = apply_spectral_perturbation(x, mask)
    
    # Energy should decrease
    in_energy = (x**2).sum()
    out_energy = (out**2).sum()
    
    print(f"In Energy: {in_energy}")
    print(f"Out Energy: {out_energy}")
    
    assert out_energy < in_energy, "Dampening should reduce energy."
    print("Perturbation Test Passed.")

if __name__ == "__main__":
    test_rapsd()
    test_mask()
    test_perturbation()
