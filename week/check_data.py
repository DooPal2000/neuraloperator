"""
Quick script to check Navier-Stokes data properties
"""

import torch
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from pathlib import Path

# Load a small sample
train_loader, test_loaders, data_processor = load_navier_stokes_pt(
    data_root=Path("~/data/navier_stokes/").expanduser(),
    train_resolution=128,
    n_train=10,
    batch_size=4,
    test_resolutions=[128],
    n_tests=[5],
    test_batch_sizes=[4],
    encode_input=True,
    encode_output=True,
)

# Check raw sample from dataset
sample = train_loader.dataset[0]
print("Raw sample keys:", sample.keys())
print("x type:", type(sample["x"]))
print("x shape:", sample["x"].shape if hasattr(sample["x"], "shape") else "N/A")

# Check preprocessed sample
sample_proc = data_processor.preprocess(sample, batched=False)
print("\nPreprocessed sample keys:", sample_proc.keys())
print("x type:", type(sample_proc["x"]))
print(
    "x shape:", sample_proc["x"].shape if hasattr(sample_proc["x"], "shape") else "N/A"
)

if hasattr(sample_proc["x"], "min"):
    print("x range:", float(sample_proc["x"].min()), "-", float(sample_proc["x"].max()))
    print("x mean:", float(sample_proc["x"].mean()))
    print("x std:", float(sample_proc["x"].std()))

if hasattr(sample_proc["y"], "min"):
    print(
        "\ny shape:",
        sample_proc["y"].shape if hasattr(sample_proc["y"], "shape") else "N/A",
    )
    print("y range:", float(sample_proc["y"].min()), "-", float(sample_proc["y"].max()))
    print("y mean:", float(sample_proc["y"].mean()))
    print("y std:", float(sample_proc["y"].std()))

# Check a batch
batch = next(iter(train_loader))
print("\nBatch keys:", batch.keys())
print("x type:", type(batch["x"]))
print("x shape:", batch["x"].shape)
print("y type:", type(batch["y"]))
print("y shape:", batch["y"].shape)
