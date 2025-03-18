# test_setup.py
import torch
import trimesh
import numpy as np

# Check for Metal support
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    # Create small test tensor
    x = torch.rand(3, 3, device=device)
    print(f"Test tensor on MPS: {x}")
    
# Test trimesh
mesh = trimesh.creation.box()
print(f"Trimesh test: Created mesh with {len(mesh.faces)} triangles")

print("Setup verification complete!")