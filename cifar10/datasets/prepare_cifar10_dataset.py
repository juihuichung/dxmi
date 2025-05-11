import os
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.datasets import CIFAR10
from torch_fidelity import calculate_metrics
import numpy as np

# CHANGE THIS to your preferred scratch path
SCRATCH_PATH = "/scratch/gpfs/cz5047/cifar10_dataset"
IMAGE_DIR = os.path.join(SCRATCH_PATH, "cifar10_train_png")
FID_STATS_PATH = os.path.join(SCRATCH_PATH, "cifar10_train_fid_stats.pt")

def download_and_save_images():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    dataset = CIFAR10(root=SCRATCH_PATH, train=True, download=True)

    print(f"Saving {len(dataset)} images to '{IMAGE_DIR}'...")
    for idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
        img_path = os.path.join(IMAGE_DIR, f"{label}_{idx:05d}.png")
        img.save(img_path)

    print("✅ Done saving PNGs.")

def compute_fid_stats():
    print("📊 Computing FID stats using torch-fidelity...")
    metrics = calculate_metrics(
        input1=IMAGE_DIR,
        isc=False,
        fid=True,
        kid=False,
    )

    fid_score = metrics['frechet_inception_distance']
    torch.save(fid_score, FID_STATS_PATH)
    print(f"✅ Saved FID score ({fid_score:.4f}) to '{FID_STATS_PATH}'.")


# from torch_fidelity.helpers import create_feature_extractor
# from torch_fidelity.calculate_metrics import _compute_statistics_of_path
import torch

def prepare_fid_stats():
    print("📊 Extracting FID stats (no comparison)...")

    stats = _compute_statistics_of_path(
        path=IMAGE_DIR,
        feature_extractor=create_feature_extractor(name='inception-v3-compat'),
        capture_all=True,
        mean_std=True
    )

    torch.save({
        'mu': stats['mu'],
        'sigma': stats['sigma']
    }, FID_STATS_PATH)

    print(f"✅ Saved FID stats to '{FID_STATS_PATH}'")

def convert_fid():
    # Path to your .npz file
    npz_path = "datasets/fid_stats_cifar10_train.npz"
    pt_path = "datasets/cifar10_train_fid_stats.pt"  # Output .pt file

    # Load .npz contents
    data = np.load(npz_path)
    mu = data["mu"]
    sigma = data["sigma"]

    # Save as .pt format
    torch.save({"mu": torch.tensor(mu), "sigma": torch.tensor(sigma)}, pt_path)
    print(f"✅ Saved: {pt_path}")

def check_valid():

    pt_path = "datasets/cifar10_train_fid_stats.pt"

    # Load the .pt file
    stats = torch.load(pt_path)

    # Check keys
    assert "mu" in stats and "sigma" in stats, "Missing keys 'mu' and/or 'sigma'"
    print("✅ Keys exist: 'mu' and 'sigma'")

    # Check shapes
    mu, sigma = stats["mu"], stats["sigma"]
    assert mu.shape == (2048,), f"Unexpected mu shape: {mu.shape}"
    assert sigma.shape == (2048, 2048), f"Unexpected sigma shape: {sigma.shape}"
    print(f"✅ mu shape: {mu.shape}")
    print(f"✅ sigma shape: {sigma.shape}")

    # Check types
    assert isinstance(mu, torch.Tensor), "mu is not a tensor"
    assert isinstance(sigma, torch.Tensor), "sigma is not a tensor"
    print("✅ Both values are PyTorch tensors")

    print("🎉 Your FID stats file is valid and ready to use!")



if __name__ == "__main__":
    # download_and_save_images()
    # compute_fid_stats()
    # prepare_fid_stats()
    # convert_fid()
    check_valid()
