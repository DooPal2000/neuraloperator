import torch

train_data = torch.load("../week5/data/navier_stokes_week5/processed/navier_stokes_week5_train.pt",
                        map_location="cpu", weights_only=False)

nu_channel = train_data["x"][:, 1, 0, 0]  # 각 샘플의 nu 값 (scalar이므로 [0,0] 위치)
print("전체 unique nu 값:", nu_channel.unique())
print("nu 분포 샘플 수:", {f"{v:.4f}": (nu_channel == v).sum().item() for v in nu_channel.unique()})