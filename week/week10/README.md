# Week 10: Gradio UI Demo

This directory contains an interactive Gradio web interface for the Neural Operator model trained in Week 8.

## Setup

Activate the conda environment:

```bash
conda activate fno_no_example
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Demo

```bash
python week10_gradio_ui.py
```

This will launch a local web server with the Gradio interface. The URL will be displayed in the console, and a public share link will also be provided.

## Features

### Interactive Controls
- **Model Selector**: Toggle between TFNO (compressed) and FNO (baseline) architectures
- **Sliders**: Adjust model parameters in real-time
  - Fourier Modes (8-64): Controls frequency domain resolution
  - Hidden Channels (16-128): Controls model capacity
- **Random Input**: Generate random input fields for quick testing
- **Clear**: Reset all inputs

### Input Handling
- Upload vorticity and viscosity field images (grayscale)
- Support for upload, clipboard, and webcam sources
- Drag and drop images

### Visualization
- Side-by-side comparison of input fields and predictions
- Image comparison slider (drag left/right to compare)
- Color-coded plots with colorbars
- Downloadable prediction results

### Examples
- Pre-loaded example inputs for quick start

## Model

The interface supports two model architectures:

### TFNO (Tensorized FNO)
- n_modes: (32, 32) [adjustable]
- hidden_channels: 64 [adjustable]
- in_channels: 2 (vorticity + viscosity)
- out_channels: 1 (next vorticity)
- factorization: "tucker"
- rank: 0.05 (5-10% parameter compression)

### FNO (Baseline)
- n_modes: (32, 32) [adjustable]
- hidden_channels: 64 [adjustable]
- in_channels: 2 (vorticity + viscosity)
- out_channels: 1 (next vorticity)

## Checkpoints

The UI automatically loads the latest checkpoint from `checkpoints/week8/`. If no checkpoint is found, the models are initialized with random weights.

## Notes

- Input images should be grayscale and properly sized (typically 64x64 or similar)
- The model expects normalized inputs, which are handled automatically
- For best results, use images from the Navier-Stokes dataset or similar fluid flow simulations
- Adjust sliders to see how model parameters affect predictions
- Use the comparison slider to visually assess prediction quality