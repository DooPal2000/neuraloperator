"""
Week 5: train_week5.py
Train FNO on custom variable-viscosity Navier-Stokes dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import logging
import sys
import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset
from neuralop import H1Loss, LpLoss
from neuralop.models import FNO, TFNO
from neuralop.training import Trainer
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer


# ── 로거 설정 (파일 + 콘솔 동시 출력) ──────────────────────────
log_dir = Path("results/week8")
log_dir.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_dir / f"train_log_{ts}.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),   # 콘솔에도 출력
    ],
)
logger = logging.getLogger(__name__)

# 기존 import들 아래에 추가
warnings.filterwarnings("ignore", message=".*received unexpected keyword arguments.*")



def load_processed_dataset(root="../week5/data/navier_stokes_week5/processed"):
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

    # model = FNO(
    #     n_modes=(32, 32),
    #     hidden_channels=64,
    #     in_channels=2,
    #     out_channels=1,
    # ).to(device)

    model = TFNO(
        n_modes=(32, 32),
        hidden_channels=64,
        in_channels=2,
        out_channels=1,
        factorization="tucker",
        implementation="factorized",
        rank=0.05,  # 파라미터 약 5~10% 수준으로 압축
    ).to(device)

    # RESUME_CHECKPOINT = "checkpoints/week5/fno_week5_20260418_023715.pt"  # 경로 지정
    RESUME_CHECKPOINT = ""
    if RESUME_CHECKPOINT:
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from: {RESUME_CHECKPOINT}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    trainer = Trainer(
        model=model,
        n_epochs=1,
        device=device,
        data_processor=data_processor,
        eval_interval=1,
        use_distributed=False,
        verbose=False,
        log_output=False,
        mixed_precision=False,
    )

    train_loss = H1Loss(d=2)
    eval_losses = {
        "h1": H1Loss(d=2),
        "l2": LpLoss(d=2, p=2),
    }

    # history = trainer.train(
    #     train_loader=train_loader,
    #     test_loaders={128: test_loader},
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     regularizer=False,
    #     training_loss=train_loss,
    #     eval_losses=eval_losses,
    # )

    # print("▶ history keys:", list(history.keys()))
    # print("▶ history 내용:", history)

    train_loss_history = []
    eval_h1_history    = []
    eval_l2_history    = []

    logger.info(f"{'Epoch':>6} | {'train_loss':>10} | {'val_h1':>8} | {'val_l2':>8} | {'time(s)':>8}")
    logger.info("-" * 52)

    for epoch in range(50):
        t0 = time.time()
        
        history = trainer.train(
            train_loader=train_loader,
            test_loaders={128: test_loader},
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=False,
            training_loss=train_loss,
            eval_losses=eval_losses,
        )
        elapsed = time.time() - t0

        tl = float(history["avg_loss"])
        h1 = float(history["128_h1"])
        l2 = float(history["128_l2"])
        
        train_loss_history.append(tl)
        eval_h1_history.append(h1)
        eval_l2_history.append(l2)

        logger.info(f"[{epoch+1:3d}/50] | {tl:10.4f} | {h1:8.4f} | {l2:8.4f} | {elapsed:8.2f}s")


    ckpt_dir = Path("checkpoints/week8")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = ckpt_dir / f"fno_week8_{ts}.pt"

    config = {
        "n_modes": (32, 32),
        "hidden_channels": 64,
        "in_channels": 2,
        "out_channels": 1,
        "n_epochs": 6,
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

    results_dir = Path("results/week8")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_results = {
        "config": config,
        "timestamp": ts,
        "checkpoint_path": str(save_path),
        "train_loss_history": train_loss_history,    # 리스트 저장
        "eval_history": {"128": {"h1": eval_h1_history, "l2": eval_l2_history}},
    }
    with open(results_dir / "train_results.json", "w", encoding="utf-8") as f:
        json.dump(train_results, f, indent=2)

    with open(results_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"saved checkpoint: {save_path}")
    print(f"saved results: {results_dir / 'train_results.json'}")


if __name__ == "__main__":
    main()
