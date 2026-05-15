import gradio as gr
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from neuralop import TFNO, FNO
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from gradio_imageslider import ImageSlider


device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "TFNO (Tucker)": TFNO(
        n_modes=(32, 32),
        hidden_channels=64,
        in_channels=2,
        out_channels=1,
        factorization="tucker",
        implementation="factorized",
        rank=0.05,
    ).to(device),
    "FNO (Baseline)": FNO(
        n_modes=(32, 32),
        hidden_channels=64,
        in_channels=2,
        out_channels=1,
    ).to(device),
}

checkpoint_path = Path("checkpoints/week8")
if checkpoint_path.exists():
    ckpt_files = list(checkpoint_path.glob("*.pt"))
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        models["TFNO (Tucker)"].load_state_dict(ckpt["model_state_dict"])
        models["TFNO (Tucker)"].eval()
        models["FNO (Baseline)"].load_state_dict(ckpt["model_state_dict"])
        models["FNO (Baseline)"].eval()

for name in models:
    models[name].eval()


x_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
y_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])

data_processor = DefaultDataProcessor(
    in_normalizer=x_normalizer,
    out_normalizer=y_normalizer,
)


def predict(input_vorticity, input_viscosity, model_name, n_modes, hidden_channels):
    model = models[model_name]

    vort = input_vorticity.astype(np.float32) / 255.0
    visc = input_viscosity.astype(np.float32) / 255.0

    x = np.stack([vort, visc], axis=0)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)

    pred = pred.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(input_vorticity, cmap="RdBu", origin="lower")
    axes[0].set_title("Input Vorticity")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(input_viscosity, cmap="viridis", origin="lower")
    axes[1].set_title("Input Viscosity")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(pred[0], cmap="RdBu", origin="lower")
    axes[2].set_title("Predicted Next Vorticity")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    return fig, pred[0], (input_vorticity, pred[0])


with gr.Blocks() as demo:
    gr.Markdown("# Neural Operator Demo - Navier-Stokes")
    gr.Markdown(
        "Predict next vorticity field from current vorticity and viscosity fields."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Settings")
            model_selector = gr.Radio(
                choices=list(models.keys()),
                value="TFNO (Tucker)",
                label="Model Architecture",
                info="Choose between TFNO (compressed) and FNO (baseline)",
            )
            n_modes_slider = gr.Slider(
                minimum=8,
                maximum=64,
                value=32,
                step=8,
                label="Fourier Modes",
                info="Controls resolution in frequency domain",
            )
            hidden_channels_slider = gr.Slider(
                minimum=16,
                maximum=128,
                value=64,
                step=16,
                label="Hidden Channels",
                info="Model capacity/width",
            )

        with gr.Column(scale=2):
            gr.Markdown("### Inputs")
            with gr.Row():
                input_vorticity = gr.Image(
                    image_mode="L",
                    sources=["upload", "clipboard", "webcam"],
                    type="numpy",
                    label="Current Vorticity Field",
                    height=300,
                )
                input_viscosity = gr.Image(
                    image_mode="L",
                    sources=["upload", "clipboard", "webcam"],
                    type="numpy",
                    label="Viscosity Field",
                    height=300,
                )

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                random_btn = gr.Button("Random Input", variant="secondary")
                predict_btn = gr.Button("Predict", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Output")
            output_plot = gr.Plot(label="Prediction Result")
            output_image = gr.Image(label="Prediction Image", type="numpy")

        with gr.Column():
            gr.Markdown("### Comparison")
            comparison = ImageSlider(
                label="Drag slider to compare Input vs Prediction",
                type="numpy",
            )

    def generate_random():
        return (
            (np.random.rand(64, 64) * 255).astype(np.uint8),
            (np.random.rand(64, 64) * 255).astype(np.uint8),
        )

    def clear_inputs():
        return None, None

    predict_btn.click(
        fn=predict,
        inputs=[
            input_vorticity,
            input_viscosity,
            model_selector,
            n_modes_slider,
            hidden_channels_slider,
        ],
        outputs=[output_plot, output_image, comparison],
    )

    random_btn.click(
        fn=generate_random,
        outputs=[input_vorticity, input_viscosity],
    )

    clear_btn.click(
        fn=clear_inputs,
        outputs=[input_vorticity, input_viscosity],
    )

    gr.Markdown("### Examples")
    gr.Examples(
        examples=[
            [
                (np.random.rand(64, 64) * 255).astype(np.uint8),
                (np.random.rand(64, 64) * 255).astype(np.uint8),
            ],
            [
                (np.clip(np.random.randn(64, 64) * 50 + 127, 0, 255)).astype(np.uint8),
                (np.random.rand(64, 64) * 100).astype(np.uint8),
            ],
        ],
        inputs=[input_vorticity, input_viscosity],
    )


if __name__ == "__main__":
    demo.launch(share=True)
