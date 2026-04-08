"""
Week 3: 기본 FNO 모델 학습 (Navier-Stokes) - FIXED VERSION

Navier-Stokes 방정식에 대한 Fourier Neural Operator 학습
- Input: 초기 vorticity field (128x128)
- Output: 미래 vorticity field (128x128)
- Goal: Relative L2 error < 0.02

UPDATED HYPERPARAMETERS based on research:
- Increased n_modes from (64,64) to (128,128)
- Increased hidden_channels from 64 to 128
- Increased epochs from 20 to 50


- 재학습 코드를 수정하였습니다.
- 모델만 가지고 재학습 (실전에서 흔히 사용) 하려면,
week3_train.py 파일로 이동하여 실행

"""

from pathlib import Path
import torch
from datetime import datetime

from neuralop.models import FNO
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.training import Trainer
from neuralop import LpLoss, H1Loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path("~/data/navier_stokes/").expanduser()

    config = {
        "n_train": 1000,
        "n_tests": [100],
        "batch_size": 4,
        "test_batch_sizes": [8],
        "train_resolution": 128,
        "test_resolutions": [128],
        "n_epochs": 21,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "model_n_modes": (64, 64),
        "model_hidden_channels": 64,
    }

    print("=" * 60)
    print("FNO Training (Option 1: Fresh Optimizer)")
    print("=" * 60)

    # ========================================
    # 1. 데이터 로드
    # ========================================
    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_dir,
        train_resolution=config["train_resolution"],
        n_train=config["n_train"],
        batch_size=config["batch_size"],
        test_resolutions=config["test_resolutions"],
        n_tests=config["n_tests"],
        test_batch_sizes=config["test_batch_sizes"],
        encode_input=True,
        encode_output=True,
    )

    batch = next(iter(train_loader))
    print(f"Input shape: {batch['x'].shape}")
    print(f"Output shape: {batch['y'].shape}")

    # ========================================
    # 2. 모델 생성 + (선택) 가중치 로드
    # ========================================
    model = FNO(
        n_modes=config["model_n_modes"],
        hidden_channels=config["model_hidden_channels"],
        in_channels=1,
        out_channels=1,
    ).to(device)

    checkpoint_path = "checkpoints/week3/fno_navier_stokes_baseline_updated.pt"

    if Path(checkpoint_path).exists():
        print("Loading model weights only (Option 1)...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No checkpoint found → training from scratch")

    # ========================================
    # 3. optimizer / scheduler (항상 새로)
    # ========================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["n_epochs"],
    )

    # ========================================
    # 4. loss
    # ========================================
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    # ========================================
    # 5. Trainer
    # ========================================
    trainer = Trainer(
        model=model,
        n_epochs=config["n_epochs"],
        device=device,
        data_processor=data_processor,
        eval_interval=5,
        verbose=True,
        log_output=True,
        mixed_precision=True,  # 문제 있으면 False
    )

    # ========================================
    # 6. 학습
    # ========================================
    print("Starting training...")

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=h1loss,
        eval_losses={"h1": h1loss, "l2": l2loss},
    )

    # ========================================
    # 7. 저장
    # ========================================
    checkpoint_dir = Path("checkpoints/week3")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = checkpoint_dir / f"fno_navier_stokes_{timestamp}.pt"

    # save_path = checkpoint_dir / "fno_navier_stokes_baseline_updated.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "timestamp": timestamp,  # 체크포인트 안에도 기록
        },
        save_path,
    )

    print(f"Model saved to: {save_path}")

    return model


if __name__ == "__main__":
    main()
