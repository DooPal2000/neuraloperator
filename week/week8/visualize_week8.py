"""visualize_week8.py"""
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from neuralop.models import TFNO
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor

# ── 설정 ──────────────────────────────────────────────────────
CKPT_PATH   = "checkpoints/week8/fno_week8_20260501_143014.pt"  # ← 실제 파일명으로 교체
DATA_ROOT   = "../week5/data/navier_stokes_week5/processed"
RESULTS_DIR = Path("results/week8")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 데이터 로드 ───────────────────────────────────────────────
train_data = torch.load(f"{DATA_ROOT}/navier_stokes_week5_train.pt",
                        map_location="cpu", weights_only=False)
test_data  = torch.load(f"{DATA_ROOT}/navier_stokes_week5_test.pt",
                        map_location="cpu", weights_only=False)

x_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
y_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
x_normalizer.fit(train_data["x"])
y_normalizer.fit(train_data["y"])

data_processor = DefaultDataProcessor(
    in_normalizer=x_normalizer,
    out_normalizer=y_normalizer,
).to(device)

# ── 모델 로드 ─────────────────────────────────────────────────
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model = TFNO(
    n_modes=(32, 32), hidden_channels=64,
    in_channels=2, out_channels=1,
    factorization="tucker", implementation="factorized", rank=0.05,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── nu별로 대표 샘플 1개씩 뽑기 ──────────────────────────────
nu_vals = test_data["x"][:, 1, 0, 0]
unique_nus = nu_vals.unique()

samples = []
for nu in unique_nus:
    idx = (nu_vals == nu).nonzero(as_tuple=True)[0][0].item()
    samples.append((nu.item(), idx))

# ── Inference ─────────────────────────────────────────────────
fig, axes = plt.subplots(len(samples), 3, figsize=(12, 3 * len(samples)))
fig.suptitle("TFNO Prediction vs Ground Truth (Variable Viscosity)", fontsize=14)

with torch.no_grad():
    for row, (nu, idx) in enumerate(samples):
        x = test_data["x"][idx:idx+1].to(device)
        y_true = test_data["y"][idx:idx+1].to(device)

        sample = {"x": x, "y": y_true}
        sample = data_processor.preprocess(sample)
        out = model(sample["x"])

        out = data_processor.postprocess(out, sample)
        if isinstance(out, tuple):
            out = out[0]  # (tensor, sample) 튜플이면 tensor만 꺼냄
        pred = out.squeeze().cpu().numpy()

        true = y_true.squeeze().cpu().numpy()
        err  = np.abs(pred - true)

        vmin = min(pred.min(), true.min())
        vmax = max(pred.max(), true.max())

        ax = axes[row]
        im0 = ax[0].imshow(true, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax[0].set_title(f"ν={nu:.4f}  |  Ground Truth")
        ax[0].axis("off")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)

        im1 = ax[1].imshow(pred, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax[1].set_title("Prediction")
        ax[1].axis("off")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)

        im2 = ax[2].imshow(err, cmap="hot_r")
        ax[2].set_title(f"Abs Error  (max={err.max():.4f})")
        ax[2].axis("off")
        plt.colorbar(im2, ax=ax[2], fraction=0.046)

plt.tight_layout()
save_path = RESULTS_DIR / "prediction_vs_truth.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"saved: {save_path}")
plt.show()