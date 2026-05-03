import torch
from pathlib import Path

root = Path("../week5/data/navier_stokes_week5/processed")

train_data = torch.load(root / "navier_stokes_week5_train.pt", map_location="cpu", weights_only=False)

print("x shape:", train_data["x"].shape)           # (N, C, H, W)
print("y shape:", train_data["y"].shape)

# 2번째 채널(index 1)이 점도인지 확인
ch1 = train_data["x"][:, 1, :, :]
print("ch1 unique values (샘플당 다른가):", ch1.reshape(ch1.shape[0], -1).mean(dim=1)[:10])
print("ch1 내 공간 분산 (spatial):", ch1[0].std().item())  # 0이면 uni