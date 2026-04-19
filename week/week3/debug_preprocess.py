from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from pathlib import Path
import torch

data_dir = Path("~/data/navier_stokes/").expanduser()
train_loader, test_loaders, data_processor = load_navier_stokes_pt(
    data_root=data_dir,
    train_resolution=128,
    n_train=1000,
    batch_size=4,
    test_resolutions=[128],
    n_tests=[100],
    test_batch_sizes=[8],
    encode_input=True,
    encode_output=True,
)

# 데이터 샘플 확인
import torch

train_loader.dataset.num_workers = 0
batch = next(iter(train_loader))
print("=== Original Batch ===")
print("Keys:", batch.keys())
print("x shape:", batch["x"].shape)
print("y shape:", batch["y"].shape)
print("x dtype:", batch["x"].dtype)
print("y dtype:", batch["y"].dtype)
print("x range:", batch["x"].min().item(), "-", batch["x"].max().item())
print("y range:", batch["y"].min().item(), "-", batch["y"].max().item())

# preprocess 확인
print("\n=== Preprocessed Batch ===")
batch_proc = data_processor.preprocess(batch, batched=True)
print("Type:", type(batch_proc))
if isinstance(batch_proc, dict):
    print("Keys:", batch_proc.keys())
    print("x shape:", batch_proc["x"].shape)
    print("x dtype:", batch_proc["x"].dtype)
    print("x range:", batch_proc["x"].min().item(), "-", batch_proc["x"].max().item())
    if "y" in batch_proc:
        print("y shape:", batch_proc["y"].shape)
        print("y dtype:", batch_proc["y"].dtype)
        print(
            "y range:", batch_proc["y"].min().item(), "-", batch_proc["y"].max().item()
        )
else:
    print("batch_proc:", batch_proc)
