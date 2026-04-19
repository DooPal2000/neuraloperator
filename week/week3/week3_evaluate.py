"""
Week 3: FNO Evaluation - Navier-Stokes
Trainer.evaluate 기반 최종본
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuralop import H1Loss, LpLoss
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.models import FNO
from neuralop.training import Trainer


def select_checkpoint():
    checkpoint_dir = Path("checkpoints/week3")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"디렉토리 없음: {checkpoint_dir}")

    candidates = sorted(checkpoint_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError("checkpoints/week3/ 에 .pt 파일이 없습니다.")

    print("=" * 60)
    for idx, path in enumerate(candidates):
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"  [{idx}] {path.name}  ({size_mb:.1f} MB)")
    print(f"  [Enter] 기본값: {candidates[-1].name}  (최신)")
    print("=" * 60)

    while True:
        user_input = input("번호 입력 (Enter = 최신): ").strip()
        if user_input == "":
            return candidates[-1]
        if user_input.isdigit() and int(user_input) < len(candidates):
            return candidates[int(user_input)]
        print("  ✗ 잘못된 입력.")


def load_model_only(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = FNO(
        n_modes=config["model_n_modes"],
        hidden_channels=config["model_hidden_channels"],
        in_channels=1,
        out_channels=1,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config


def build_eval_objects(config, data_dir, device):
    _, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_dir,
        train_resolution=config["train_resolution"],
        n_train=config["n_train"],
        n_tests=config["n_tests"],
        batch_size=32,
        test_batch_sizes=[32],
        test_resolutions=config["test_resolutions"],
        encode_input=True,
        encode_output=True,
    )

    test_resolution = config["test_resolutions"][0]
    test_loader = test_loaders[test_resolution]

    trainer = Trainer(
        model=None,  # 아래에서 교체
        n_epochs=1,
        device=device,
        data_processor=data_processor,
        eval_interval=1,
        verbose=False,
        log_output=False,
        mixed_precision=False,
    )

    return test_loader, data_processor, trainer


def run_trainer_evaluate(model, trainer, test_loader, test_resolution):
    trainer.model = model
    trainer.data_processor = trainer.data_processor  # 명시적 유지

    losses = {
        "h1": H1Loss(d=2),
        "l2": LpLoss(d=2, p=2),
    }

    metrics = trainer.evaluate(
        loss_dict=losses,
        data_loader=test_loader,
        log_prefix=str(test_resolution),
        epoch=None,
        mode="single_step",
    )
    return metrics


def measure_inference_time(model, test_loader, data_processor, device, n_samples=10):
    is_cuda = device == "cuda"
    times = []

    model.eval()
    data_processor = data_processor.to(device)
    data_processor.eval()

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            if i >= n_samples:
                break

            sample = data_processor.preprocess(sample)
            x = sample["x"]

            if i == 0:
                for _ in range(3):
                    _ = model(x)

            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return float(np.mean(times)), float(np.std(times))


def visualize_predictions(model, test_loader, data_processor, device, n_samples=3, save_path=None):
    model.eval()
    data_processor = data_processor.to(device)
    data_processor.eval()

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i in range(n_samples):
            sample = test_loader.dataset[i]
            sample = data_processor.preprocess(sample, batched=False)

            x = sample["x"]
            if x.dim() == 3:
                x = x.unsqueeze(0)

            pred = model(x)
            pred_out, sample_out = data_processor.postprocess(pred, sample)

            y_true = sample_out["y"].detach().cpu().numpy().squeeze()
            y_pred = pred_out.detach().cpu().numpy().squeeze()
            error_map = np.abs(y_pred - y_true)

            vmin = min(y_true.min(), y_pred.min())
            vmax = max(y_true.max(), y_pred.max())

            im0 = axes[i, 0].imshow(y_true, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
            im1 = axes[i, 1].imshow(y_pred, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
            im2 = axes[i, 2].imshow(error_map, cmap="hot", origin="lower")

            axes[i, 0].set_title(f"Ground Truth (sample {i})")
            axes[i, 1].set_title(f"Prediction (sample {i})")
            axes[i, 2].set_title(f"Absolute Error (sample {i})")
            axes[i, 2].set_xlabel(f"Max: {error_map.max():.4f}  Mean: {error_map.mean():.4f}")

            plt.colorbar(im0, ax=axes[i, 0])
            plt.colorbar(im1, ax=axes[i, 1])
            plt.colorbar(im2, ax=axes[i, 2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 시각화 저장: {save_path}")
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("~/data/navier_stokes/").expanduser()

    print(f"Device: {device}\n")

    checkpoint_path = select_checkpoint()
    print(f"\n✓ 선택: {checkpoint_path}\n")

    print("=" * 60)
    print("Week 3: Baseline Performance Evaluation")
    print("=" * 60)

    model, config = load_model_only(checkpoint_path, device)
    print(f"config: {config}\n")

    print("Loading test dataset...")
    test_loader, data_processor, trainer = build_eval_objects(config, data_dir, device)
    test_resolution = config["test_resolutions"][0]
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print("✓ data_processor 출처: newly created from dataset config\n")

    trainer.model = model

    print("Computing metrics...")
    metrics = run_trainer_evaluate(model, trainer, test_loader, test_resolution)

    l2_mean = float(metrics[f"{test_resolution}_l2"])
    h1_mean = float(metrics[f"{test_resolution}_h1"])

    print()
    print("=" * 60)
    print("Performance Metrics")
    print("=" * 60)
    print(f"  Relative L2 : {l2_mean:.4f}")
    print(f"  H1 Error    : {h1_mean:.4f}")

    target = 0.02
    print()
    if l2_mean < target:
        print(f"✓ 목표 달성! L2 {l2_mean:.4f} < {target}")
    else:
        print(f"✗ 목표 미달 (L2={l2_mean:.4f} >= {target})")
        print("  → n_epochs / n_modes / hidden_channels 증가 고려")

    print()
    print("Measuring inference time...")
    t_mean, t_std = measure_inference_time(model, test_loader, data_processor, device)
    print(f"✓ 평균 추론 시간 : {t_mean:.4f} ± {t_std:.4f} s")
    print(f"✓ Throughput     : {1 / t_mean:.1f} batches/s")

    print()
    print("Generating visualizations...")
    viz_dir = Path("visualizations/week3")
    viz_dir.mkdir(parents=True, exist_ok=True)
    visualize_predictions(
        model,
        test_loader,
        data_processor,
        device,
        n_samples=3,
        save_path=viz_dir / "baseline_predictions.png",
    )

    results = {
        "l2_mean": l2_mean,
        "h1_mean": h1_mean,
        "inference_time_mean": t_mean,
        "inference_time_std": t_std,
        "config": config,
        "checkpoint": str(checkpoint_path),
    }

    results_dir = Path("results/week3")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "baseline_metrics.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  L2  : {l2_mean:.4f}  (target < {target})")
    print(f"  H1  : {h1_mean:.4f}")
    print(f"  Time: {t_mean:.4f} s/batch")
    print(f"  결과: {results_file}")


if __name__ == "__main__":
    main()
    
    
# """
# Week 3: Baseline 성능 측정 (Navier-Stokes)
# """

# from pathlib import Path
# import time
# import json
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from neuralop.models import FNO
# from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
# from neuralop import LpLoss, H1Loss


# def select_checkpoint():
#     checkpoint_dir = Path("checkpoints/week3")
#     if not checkpoint_dir.exists():
#         raise FileNotFoundError(f"디렉토리 없음: {checkpoint_dir}")
#     candidates = sorted(checkpoint_dir.glob("*.pt"))
#     if not candidates:
#         raise FileNotFoundError("checkpoints/week3/ 에 .pt 파일이 없습니다.")

#     print("=" * 60)
#     print("사용 가능한 체크포인트")
#     print("=" * 60)
#     for idx, path in enumerate(candidates):
#         size_mb = path.stat().st_size / (1024 ** 2)
#         print(f"  [{idx}] {path.name}  ({size_mb:.1f} MB)")
#     print(f"  [Enter] 기본값: {candidates[-1].name}  (최신)")
#     print("=" * 60)

#     while True:
#         user_input = input("번호 입력 (Enter = 최신): ").strip()
#         if user_input == "":
#             selected = candidates[-1]
#             break
#         if user_input.isdigit() and int(user_input) < len(candidates):
#             selected = candidates[int(user_input)]
#             break
#         print(f"  ✗ 잘못된 입력. 0 ~ {len(candidates)-1} 사이 숫자를 입력하세요.")

#     print(f"\n✓ 선택: {selected}\n")
#     return selected


# def load_trained_model(checkpoint_path, device):
#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     config = checkpoint["config"]

#     model = FNO(
#         n_modes=config["model_n_modes"],
#         hidden_channels=config["model_hidden_channels"],
#         in_channels=1,
#         out_channels=1,
#     ).to(device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()

#     # ← data_processor도 함께 로드
#     data_processor = checkpoint.get("data_processor", None)
#     if data_processor is None:
#         print("⚠️  WARNING: data_processor not found in checkpoint!")

#     return model, config, data_processor

# def evaluate(model, test_loader, data_processor, device):
#     l2loss = LpLoss(d=2, p=2)
#     h1loss = H1Loss(d=2)
#     l2_errors, h1_errors = [], []

#     model.eval()
#     with torch.no_grad():
#         for batch in test_loader:
#             # batch = data_processor.preprocess(batch, batched=True)
#             x     = batch["x"].to(device)
#             y     = batch["y"].to(device)  # ← encoded y, postprocess 없음
#             pred  = model(x)

#             l2_errors.append(l2loss(pred, y).item())
#             h1_errors.append(h1loss(pred, y).item())

#     return {
#         "l2_mean": float(np.mean(l2_errors)),
#         "l2_std":  float(np.std(l2_errors)),
#         "h1_mean": float(np.mean(h1_errors)),
#         "h1_std":  float(np.std(h1_errors)),
#     }

# def measure_inference_time(model, data_loader, device, n_samples=10):
#     """순수 추론 속도만 측정 (postprocess 없음)"""
#     is_cuda = device == "cuda"
#     times = []

#     model.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(data_loader):
#             if i >= n_samples:
#                 break

#             x = batch["x"].to(device)

#             if i == 0:
#                 for _ in range(3):
#                     model(x)

#             if is_cuda:
#                 torch.cuda.synchronize()
#             t0 = time.perf_counter()
#             model(x)
#             if is_cuda:
#                 torch.cuda.synchronize()
#             times.append(time.perf_counter() - t0)

#     return float(np.mean(times)), float(np.std(times))


# def visualize_predictions(model, test_loader, data_processor, device, n_samples=3, save_path=None):
#     model.eval()
#     fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
#     if n_samples == 1:
#         axes = axes[np.newaxis, :]

#     with torch.no_grad():
#         for i in range(n_samples):
#             sample = test_loader.dataset[i]
#             sample = data_processor.preprocess(sample, batched=False)

#             x = sample["x"]
#             if x.dim() == 3:
#                 x = x.unsqueeze(0)
#             x = x.to(device)

#             y_true = sample["y"]
#             y_true_np = (y_true.cpu().numpy() if isinstance(y_true, torch.Tensor)
#                          else np.array(y_true))

#             pred     = model(x)
#             y_pred_np = pred.squeeze(0).cpu().numpy()

#             y_true_2d = y_true_np.squeeze()
#             y_pred_2d = y_pred_np.squeeze()
#             error_map = np.abs(y_pred_2d - y_true_2d)

#             vmin = min(y_true_2d.min(), y_pred_2d.min())
#             vmax = max(y_true_2d.max(), y_pred_2d.max())

#             im0 = axes[i, 0].imshow(y_true_2d,  cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
#             im1 = axes[i, 1].imshow(y_pred_2d,  cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
#             im2 = axes[i, 2].imshow(error_map,  cmap="hot",    origin="lower")

#             axes[i, 0].set_title(f"Ground Truth (sample {i})")
#             axes[i, 1].set_title(f"Prediction (sample {i})")
#             axes[i, 2].set_title(f"Absolute Error (sample {i})")
#             axes[i, 2].set_xlabel(f"Max: {error_map.max():.4f}  Mean: {error_map.mean():.4f}")

#             plt.colorbar(im0, ax=axes[i, 0])
#             plt.colorbar(im1, ax=axes[i, 1])
#             plt.colorbar(im2, ax=axes[i, 2])

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"✓ 시각화 저장: {save_path}")
#     return fig


# def main():
#     device   = "cuda" if torch.cuda.is_available() else "cpu"
#     data_dir = Path("~/data/navier_stokes/").expanduser()
#     print(f"Device: {device}\n")

#     checkpoint_path = select_checkpoint()

#     print("=" * 60)
#     print("Week 3: Baseline Performance Evaluation")
#     print("=" * 60)

#     model, config, data_processor_ckpt = load_trained_model(checkpoint_path, device)
#     print(f"✓ 모델 로드: n_modes={config['model_n_modes']}, hidden={config['model_hidden_channels']}\n")

#     # ← 항상 loader는 새로 만듦
#     print("Loading test dataset...")
#     _, test_loaders, data_processor_new = load_navier_stokes_pt(
#         data_root=data_dir,
#         train_resolution=128,
#         n_train=1000,
#         n_tests=[100],
#         batch_size=32,
#         test_batch_sizes=[32],
#         test_resolutions=[128],
#         encode_input=True,
#         encode_output=True,
#     )
#     test_loader = test_loaders[128]
    
#     batch = next(iter(test_loader))
#     print(f"x mean/std: {batch['x'].mean():.4f} / {batch['x'].std():.4f}")
#     print(f"y mean/std: {batch['y'].mean():.4f} / {batch['y'].std():.4f}")
    
#     print(f"✓ Test samples: {len(test_loader.dataset)}\n")

#     # checkpoint에 data_processor 있으면 그걸 쓰고, 없으면 새로 만든 것 사용
#     data_processor = data_processor_ckpt if data_processor_ckpt is not None else data_processor_new
#     print(f"data_processor 출처: {'checkpoint' if data_processor_ckpt is not None else 'new (fallback)'}")

#     # 성능 측정
#     print("Computing metrics...")
#     metrics = evaluate(model, test_loader, data_processor, device)

#     print()
#     print("=" * 60)
#     print("Performance Metrics")
#     print("=" * 60)
#     print(f"  Relative L2 : {metrics['l2_mean']:.4f} ± {metrics['l2_std']:.4f}")
#     print(f"  H1 Error    : {metrics['h1_mean']:.4f} ± {metrics['h1_std']:.4f}")
#     print()

#     target = 0.02
#     if metrics["l2_mean"] < target:
#         print(f"✓ 목표 달성! L2 < {target}")
#     else:
#         print(f"✗ 목표 미달 (L2={metrics['l2_mean']:.4f} >= {target})")
#         print("  → n_epochs / n_modes / hidden_channels 증가 고려")
#     print()

#     # 추론 시간
#     print("Measuring inference time...")
#     t_mean, t_std = measure_inference_time(model, test_loader, device)
#     print(f"✓ 평균 추론 시간 : {t_mean:.4f} ± {t_std:.4f} s")
#     print(f"✓ Throughput     : {1/t_mean:.1f} batches/s\n")

#     # 시각화
#     print("Generating visualizations...")
#     viz_dir = Path("visualizations/week3")
#     viz_dir.mkdir(parents=True, exist_ok=True)
#     visualize_predictions(
#         model, test_loader, data_processor, device,
#         n_samples=3,
#         save_path=viz_dir / "baseline_predictions.png",
#     )

#     # 결과 저장
#     results = {
#         **metrics,
#         "inference_time_mean": t_mean,
#         "inference_time_std":  t_std,
#         "config": config,
#     }
#     results_dir = Path("results/week3")
#     results_dir.mkdir(parents=True, exist_ok=True)
#     results_file = results_dir / "baseline_metrics.json"
#     with open(results_file, "w") as f:
#         json.dump(results, f, indent=2, default=str)

#     print()
#     print("=" * 60)
#     print("Summary")
#     print("=" * 60)
#     print(f"  L2  : {metrics['l2_mean']:.4f}  (target < {target})")
#     print(f"  H1  : {metrics['h1_mean']:.4f}")
#     print(f"  Time: {t_mean:.4f} s/batch")
#     print(f"  결과: {results_file}")


# if __name__ == "__main__":
#     main()