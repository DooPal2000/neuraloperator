"""
Week 3: Baseline 성능 측정 (Navier-Stokes) - FIXED VERSION

학습된 FNO 모델의 성능 측정:
- Relative L2 error
- H1 error
- Inference time
- Ground truth vs Prediction 비교
"""

from pathlib import Path
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from neuralop.models import FNO
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop import LpLoss, H1Loss


def select_checkpoint():
    checkpoint_dir = Path("checkpoints/week3")

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"디렉토리 없음: {checkpoint_dir}")

    candidates = sorted(checkpoint_dir.glob("*.pt"))

    if not candidates:
        raise FileNotFoundError("checkpoints/week3/ 에 .pt 파일이 없습니다.")

    print("=" * 60)
    print("사용 가능한 체크포인트")
    print("=" * 60)
    for idx, path in enumerate(candidates):
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"  [{idx}] {path.name}  ({size_mb:.1f} MB)")
    print(f"  [Enter] 기본값: {candidates[-1].name}  (최신)")
    print("=" * 60)

    while True:
        user_input = input("번호 입력 (Enter = 최신): ").strip()

        if user_input == "":
            selected = candidates[-1]
            break

        if user_input.isdigit() and int(user_input) < len(candidates):
            selected = candidates[int(user_input)]
            break

        print(f"  ✗ 잘못된 입력입니다. 0 ~ {len(candidates)-1} 사이 숫자를 입력하세요.")

    print(f"\n✓ 선택된 체크포인트: {selected}\n")
    return selected


def load_trained_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FNO(
        n_modes=config["model_n_modes"],
        hidden_channels=config["model_hidden_channels"],
        in_channels=1,
        out_channels=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


# ✅ FIX: device 문자열 대신 torch.cuda.is_available() 기반으로 sync
def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_inference_time(model, data_loader, device, n_samples=10):
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= n_samples:
                break

            x = batch["x"].to(device)

            # Warmup (첫 배치에서만)
            if i == 0:
                for _ in range(5):
                    _ = model(x)

            _cuda_sync()
            start = time.time()
            _ = model(x)
            _cuda_sync()
            end = time.time()

            times.append(end - start)

    return np.mean(times), np.std(times)


def visualize_predictions(
    model, data_loader, data_processor, device, n_samples=3, save_path=None
):
    model.eval()

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    # ✅ FIX: axes가 1D가 될 수 있으므로 항상 2D로 보장
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i in range(n_samples):
            sample = data_loader.dataset[i]
            sample = data_processor.preprocess(sample, batched=False)

            x = sample["x"]
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.to(device)

            y_true = sample["y"]
            if isinstance(y_true, torch.Tensor):
                y_true_np = y_true.cpu().numpy()
            else:
                y_true_np = np.array(y_true)

            pred = model(x)
            y_pred_np = pred.squeeze(0).cpu().numpy()

            # ✅ FIX: squeeze()로 확실하게 2D 확보 후 단일 error_map만 사용
            y_true_2d = y_true_np.squeeze()
            y_pred_2d = y_pred_np.squeeze()
            error_map = np.abs(y_pred_2d - y_true_2d)

            vmin = min(y_true_2d.min(), y_pred_2d.min())
            vmax = max(y_true_2d.max(), y_pred_2d.max())

            im0 = axes[i, 0].imshow(y_true_2d, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f"Ground Truth (sample {i})")
            plt.colorbar(im0, ax=axes[i, 0])

            im1 = axes[i, 1].imshow(y_pred_2d, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f"Prediction (sample {i})")
            plt.colorbar(im1, ax=axes[i, 1])

            im2 = axes[i, 2].imshow(error_map, cmap="hot", origin="lower")
            axes[i, 2].set_title(f"Absolute Error (sample {i})")
            plt.colorbar(im2, ax=axes[i, 2])
            axes[i, 2].set_xlabel(f"Max: {error_map.max():.4f}, Mean: {error_map.mean():.4f}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Visualization saved to: {save_path}")

    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path("~/data/navier_stokes/").expanduser()
    checkpoint_path = select_checkpoint()

    print("=" * 60)
    print("Week 3: Baseline Performance Evaluation (FIXED)")
    print("=" * 60)
    print()
    print(f"Loading model from: {checkpoint_path}")
    model, config = load_trained_model(checkpoint_path, device)

    print("Loading test dataset...")
    _, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_dir,
        train_resolution=128,
        n_train=1000,
        n_tests=[100],
        batch_size=32,
        test_batch_sizes=[32],
        test_resolutions=[128],
        encode_input=True,
        encode_output=True,
    )

    test_loader = test_loaders[128]
    print(f"✓ Test dataset size: {len(test_loader.dataset)}")
    print()

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    print("Computing performance metrics...")

    model.eval()
    l2_errors = []
    h1_errors = []

    with torch.no_grad():
        for batch in test_loader:
            batch = data_processor.preprocess(batch, batched=True)
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            # ── 디버그 ──
            print(f"x    | shape: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")
            print(f"y    | shape: {y.shape}, mean: {y.mean():.4f}, std: {y.std():.4f}")
            print(f"pred | shape: {pred.shape}, mean: {pred.mean():.4f}, std: {pred.std():.4f}")
            print(f"l2 this batch: {l2loss(pred, y).item():.4f}")
            break  # 한 배치만
            

            # # postprocess 후 두 텐서 모두 동일 device로 명시 이동
            # pred_out, out_dict = data_processor.postprocess(pred, batch)
            # y_out = out_dict["y"]

            # pred_out = pred_out.to(device)
            # y_out = y_out.to(device)

            # l2_errors.append(l2loss(pred_out, y_out).item())
            # h1_errors.append(h1loss(pred_out, y_out).item())
            
            # ✅ 변경 코드 (postprocess 없음, 정규화 공간) → 학습 중 수치와 일치
            l2_errors.append(l2loss(pred, y).item())
            h1_errors.append(h1loss(pred, y).item())


    avg_l2_error = np.mean(l2_errors)
    std_l2_error = np.std(l2_errors)
    avg_h1_error = np.mean(h1_errors)
    std_h1_error = np.std(h1_errors)

    print()
    print("=" * 60)
    print("Performance Metrics")
    print("=" * 60)
    print(f"Relative L2 Error: {avg_l2_error:.4f} ± {std_l2_error:.4f}")
    print(f"H1 Error:          {avg_h1_error:.4f} ± {std_h1_error:.4f}")
    print()

    target_error = 0.02
    if avg_l2_error < target_error:
        print(f"✓ Target achieved! L2 error < {target_error}")
    else:
        print(f"✗ Target not met. L2 error >= {target_error}")
        print("  Suggestions:")
        print("  - Increase n_epochs")
        print("  - Increase n_modes or hidden_channels")
        print("  - Increase training dataset size")
    print()

    print("Measuring inference time...")
    mean_time, std_time = measure_inference_time(model, test_loader, device, n_samples=10)
    print(f"✓ Average inference time: {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"✓ Throughput: {1 / mean_time:.2f} samples/second")
    print()

    print("Generating visualizations...")
    viz_dir = Path("visualizations/week3")
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_path = viz_dir / "baseline_predictions.png"

    visualize_predictions(
        model, test_loader, data_processor, device, n_samples=3, save_path=viz_path
    )
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Model: FNO(n_modes={config['model_n_modes']}, hidden={config['model_hidden_channels']})")
    print(f"Training: {config['n_epochs']} epochs, {config['n_train']} samples")
    print(f"L2 Error: {avg_l2_error:.4f} (target: < {target_error})")
    print(f"H1 Error: {avg_h1_error:.4f}")
    print(f"Inference: {mean_time:.4f}s per sample")
    print()

    import json

    results = {
        "l2_error": avg_l2_error,
        "l2_error_std": std_l2_error,
        "h1_error": avg_h1_error,
        "h1_error_std": std_h1_error,
        "inference_time_mean": mean_time,
        "inference_time_std": std_time,
        "config": config,
    }

    results_path = Path("results/week3")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "baseline_metrics.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Results saved to: {results_file}")
    print()


if __name__ == "__main__":
    main()