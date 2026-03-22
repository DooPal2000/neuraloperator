import matplotlib.pyplot as plt
import torch
from neuralop.data.datasets import load_navier_stokes_pt


def main():
    # 1. Load data
    print("Loading Navier-Stokes dataset...")
    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        n_train=1000,
        n_tests=[50],
        batch_size=32,
        test_batch_sizes=[32],
        train_resolution=128,
        test_resolutions=[128],
    )

    # 2. Get a sample from test set
    test_dataset = test_loaders[128].dataset
    sample = test_dataset[0]

    # 3. Preprocess the sample
    sample = data_processor.preprocess(sample, batched=False)

    # Input (initial vorticity field)
    x = sample["x"]  # Shape: [1, 128, 128]
    # Output (future vorticity field)
    y = sample["y"]  # Shape: [1, 128, 128]

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # 4. Visualize vorticity fields
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Plot multiple samples
    for i in range(3):
        sample = test_dataset[i]
        sample = data_processor.preprocess(sample, batched=False)

        x = sample["x"]
        y = sample["y"]

        # Input vorticity
        im1 = axes[0, i].imshow(x[0], cmap="RdBu_r", origin="lower")
        axes[0, i].set_title(f"Input Vorticity (sample {i})")
        plt.colorbar(im1, ax=axes[0, i])

        # Output vorticity
        # im2 = axes[1, i].imshow(y[0], cmap="RdBu_r", origin="lower")
        im2 = axes[1, i].imshow(y[0].squeeze(), cmap="RdBu_r", origin="lower")

        axes[1, i].set_title(f"Output Vorticity (sample {i})")
        plt.colorbar(im2, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig("week2_vorticity_visualization.png", dpi=150)
    print("Saved visualization to week2_vorticity_visualization.png")

    # 5. Optional: Load and visualize at different resolution
    # This requires downloading higher resolution data
    print("\nTo visualize different resolutions, uncomment the following code:")
    print("# Load data at 256 resolution if available")
    print(
        "# train_loader_256, test_loaders_256, data_processor_256 = load_navier_stokes_pt("
    )
    print("#     n_train=1000,")
    print("#     n_tests=[50],")
    print("#     batch_size=32,")
    print("#     test_batch_sizes=[32],")
    print("#     train_resolution=128,")
    print("#     test_resolutions=[256],  # 256 resolution")
    print("# )")


if __name__ == "__main__":
    main()
