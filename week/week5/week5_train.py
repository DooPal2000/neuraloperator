"""
Week 5: train_week5.py
Train FNO on custom variable-viscosity Navier-Stokes dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from neuralop import H1Loss, LpLoss
from neuralop.models import FNO
from neuralop.training import Trainer
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer


def load_processed_dataset(root="data/navier_stokes_week5/processed"):
    root = Path(root)
    train_data = torch.load(
        root / "navier_stokes_week5_train.pt", map_location="cpu", weights_only=False
    )
    test_data = torch.load(
        root / "navier_stokes_week5_test.pt", map_location="cpu", weights_only=False
    )
    return train_data, test_data


def build_loader(data_dict, batch_size=16, shuffle=False):
    dataset = TensorDataset(data_dict["x"], data_dict["y"])

    def collate(batch):
        x = torch.stack([b[0] for b in batch], dim=0)
        y = torch.stack([b[1] for b in batch], dim=0)
        return {"x": x, "y": y}

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, test_data = load_processed_dataset()

    train_loader = build_loader(train_data, batch_size=16, shuffle=True)
    test_loader = build_loader(test_data, batch_size=16, shuffle=False)

    x_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
    y_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
    x_normalizer.fit(train_data["x"])
    y_normalizer.fit(train_data["y"])

    data_processor = DefaultDataProcessor(
        in_normalizer=x_normalizer,
        out_normalizer=y_normalizer,
    )

    model = FNO(
        n_modes=(32, 32),
        hidden_channels=64,
        in_channels=2,
        out_channels=1,
    ).to(device)

    RESUME_CHECKPOINT = "checkpoints/week5/fno_week5_20260418_023715.pt"  # 경로 지정
    if RESUME_CHECKPOINT:
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from: {RESUME_CHECKPOINT}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    trainer = Trainer(
        model=model,
        n_epochs=75,
        device=device,
        data_processor=data_processor,
        eval_interval=1,
        use_distributed=False,
        verbose=True,
        log_output=False,
        mixed_precision=False,
    )

    train_loss = H1Loss(d=2)
    eval_losses = {
        "h1": H1Loss(d=2),
        "l2": LpLoss(d=2, p=2),
    }

    history = trainer.train(
        train_loader=train_loader,
        test_loaders={128: test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    ckpt_dir = Path("checkpoints/week5")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = ckpt_dir / f"fno_week5_{ts}.pt"

    config = {
        "n_modes": (32, 32),
        "hidden_channels": 64,
        "in_channels": 2,
        "out_channels": 1,
        "n_epochs": 75,
        "batch_size": 16,
        "dataset": "navier_stokes_week5",
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "timestamp": ts,
        },
        save_path,
    )

    results_dir = Path("results/week5")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_results = {
        "config": config,
        "timestamp": ts,
        "checkpoint_path": str(save_path),
        "train_loss_history": history.get("train_loss", []),
        "eval_history": history.get("eval_loss", {}),
    }

    with open(results_dir / "train_results.json", "w", encoding="utf-8") as f:
        json.dump(train_results, f, indent=2)

    with open(results_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"saved checkpoint: {save_path}")
    print(f"saved results: {results_dir / 'train_results.json'}")


if __name__ == "__main__":
    main()
