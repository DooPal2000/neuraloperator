"""
Week 5: preprocess_dataset.py
Convert raw sample files into train/test tensors for FNO training.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import torch


def stratified_split_by_nu(samples, train_ratio=0.8):
    groups = defaultdict(list)
    for s in samples:
        groups[float(s["nu"])] .append(s)

    train_samples, test_samples = [], []
    for nu, items in groups.items():
        items = sorted(items, key=lambda z: int(z["seed"]))
        n_train = max(1, int(len(items) * train_ratio))
        train_samples.extend(items[:n_train])
        test_samples.extend(items[n_train:])
    return train_samples, test_samples


def build_tensor_dataset(samples):
    xs, ys, nus, seeds = [], [], [], []
    for s in samples:
        x0 = s["x0"].float()
        y = s["y"].float()
        nu = float(s["nu"])
        nu_field = torch.full_like(x0, nu)
        x = torch.stack([x0, nu_field], dim=0)
        y = y.unsqueeze(0)

        xs.append(x)
        ys.append(y)
        nus.append(nu)
        seeds.append(int(s["seed"]))

    return {
        "x": torch.stack(xs, dim=0),
        "y": torch.stack(ys, dim=0),
        "nu": torch.tensor(nus, dtype=torch.float32),
        "seed": torch.tensor(seeds, dtype=torch.int64),
    }


def preprocess_dataset(
    input_root: str | Path = "data/navier_stokes_week5",
    output_root: str | Path | None = None,
    train_ratio: float = 0.8,
):
    input_root = Path(input_root)
    raw_dir = input_root / "raw"
    output_root = Path(output_root) if output_root is not None else input_root / "processed"
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No raw samples found in {raw_dir}")

    samples = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    train_samples, test_samples = stratified_split_by_nu(samples, train_ratio=train_ratio)

    train_data = build_tensor_dataset(train_samples)
    test_data = build_tensor_dataset(test_samples)

    torch.save(train_data, output_root / "navier_stokes_week5_train.pt")
    torch.save(test_data, output_root / "navier_stokes_week5_test.pt")

    split_info = {
        "n_total": len(samples),
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "train_ratio": train_ratio,
        "input_channels": 2,
        "output_channels": 1,
        "nu_values": sorted({float(s["nu"]) for s in samples}),
    }
    with open(output_root / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(split_info)


if __name__ == "__main__":
    preprocess_dataset()
