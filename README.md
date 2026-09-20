# CMPE-255-Data-Mining-Assignment

Course project for **CMPE 255 (Data Mining), SJSU — Fall 2025**. The project studies
Tanabe et al.'s CNN surrogate for EUV lithography simulation, and adds a Streamlit
front-end for exploring the simulator's output. It ships with the written report
(`CMPE-255 DATA MINING_Project Report.docx`) and the demo video submitted for the course.

The simulator and CNN code are the reference implementation from
[takahashi-edalab/EUVlitho](https://github.com/takahashi-edalab/EUVlitho)
(MIT, Copyright (c) 2024 Hiroyoshi Tanabe). The course artifacts are the study, the report,
the video, and the visualisation layer (`streamlit_app.py`, `analyze_data.py`,
`setup_windows.py`, `run_streamlit.bat`). Upstream's own documentation is preserved at
[`include/README.md`](include/README.md).

> The same project is also published, without the course artifacts, at
> [nitishsjsucs/EUVLithography](https://github.com/nitishsjsucs/EUVLithography). This
> repository is the original course submission.

## The problem

EUV masks are not flat. At λ = 13.5 nm with a 60 nm absorber, the mask scatters in 3D and
the thin-mask (Kirchhoff) approximation — Fourier transforming the layout — is measurably
wrong. Getting it right means solving Maxwell's equations over the mask, which is slow.
The upstream work trains a CNN to predict the 3D mask (M3D) parameters straight from the
mask bitmap, so the aerial image can be formed without the full EM solve.

## What's here

- `emint/`, `include/`, `mask/` — the rigorous EM simulator: a 3D waveguide model
  (equivalent to RCWA, fewer field components), C++ on Eigen + oneAPI MKL with MAGMA/CUDA
  for the eigensolves. Configured for λ = 13.5 nm, NA = 0.33, 4× magnification, 6° chief
  ray, dipole illumination (σ 0.55–0.9), 60 nm absorber over a 40-pair multilayer,
  2048 nm pitch at 512×512.
- `cnn/` — six PyTorch Lightning CNNs (real and imaginary parts of a0, ax, ay). Five
  conv/BatchNorm/max-pool blocks, 16→256 channels, 3×3 with **circular padding** so the
  periodic mask wraps, into a 4096-unit FC layer; regresses 1901 diffraction orders (a0)
  or 1749 (ax/ay) under MSE. Trained with pattern-shift augmentation — a random cyclic
  roll of the mask with the matching phase ramp applied analytically to the targets —
  turning 20,000 rigorous simulations into 1,000,000 examples.
- `cnnpredict/`, `cnnabbe/`, `cnnsocs/` — inference, then aerial image formation by Abbe
  summation or the STCC/SOCS formulation.
- `streamlit_app.py`, `analyze_data.py` — the added viewer and plotting script.

## Result on the shipped example

One 512×512 pattern is committed, run through every path. Differencing those CSVs
elementwise (skip each file's 4 header lines):

| Aerial image | RMSE vs rigorous EM | Max abs. error |
|---|---|---|
| CNN + Abbe (`cnnabbe/nnabbe.csv`) | **0.0061** | 0.0242 |
| CNN + SOCS (`cnnsocs/nnsocs.csv`) | **0.0059** | 0.0236 |
| Thin-mask FT (`cnnabbe/ftint.csv`) | 0.0160 | 0.0392 |

Peak EM intensity is 0.657, so the CNN sits around 0.9% of peak and is ~2.6× closer to
the rigorous result than the thin-mask approximation. This is **one pattern** from the
upstream authors, not a benchmark I ran over a test set.

## What you can and cannot run

Runnable: the Streamlit viewer and `analyze_data.py`, which plot the committed CSVs.

```bash
pip install -r streamlit_requirements.txt
streamlit run streamlit_app.py      # or run_streamlit.bat on Windows
python analyze_data.py              # writes the three output_*.png
```

Not runnable from this checkout: the EM simulator needs oneAPI, Eigen, CUDA and MAGMA and
its makefiles still carry upstream's `/home/tanabe/...` paths; the CNNs have no trained
checkpoint committed and no training set (the `.npy` files hold one sample each against
`ndata=20000` in the training scripts). The "Run CNN Inference" button in the demo tab is
a placeholder that sleeps briefly and prints a fixed message — no model is loaded.

## Attribution

Simulator and CNN from **[takahashi-edalab/EUVlitho](https://github.com/takahashi-edalab/EUVlitho)**,
MIT, © 2024 Hiroyoshi Tanabe. Primary reference:

- H. Tanabe, M. Shimode and A. Takahashi, "Rigorous electromagnetic simulator for extreme
  ultraviolet lithography and convolutional neural network reproducing electromagnetic
  simulations," *JM3* **24** (2025) 024201. https://doi.org/10.1117/1.JMM.24.2.024201

Further papers on the waveguide model, the CNN, and the augmentation strategy are listed
in [`include/README.md`](include/README.md). The illumination model in `ampS` follows
N. Davydova et al., SPIE 88860A.
