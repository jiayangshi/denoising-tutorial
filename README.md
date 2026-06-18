# Deep Learning for Image Denoising — hands-on notebooks

Four small, self-contained Colab notebooks that denoise the **same** Gaussian-noised
CelebA face with four different philosophies. Each runs in a couple of minutes on a
free Colab GPU and installs only a minimal set of packages.

| # | Notebook | Idea | Needs clean data? | Needs noise sigma? | Extra install |
|---|----------|------|:---:|:---:|---|
| 1 | [`1_BM3D.ipynb`](1_BM3D.ipynb) | Classical patch filtering (baseline) | no | **yes** | `bm3d` |
| 2 | [`2_Noise2Void.ipynb`](2_Noise2Void.ipynb) | Blind-spot self-supervision | no | no | — (PyTorch) |
| 3 | [`3_Noise2Self.ipynb`](3_Noise2Self.ipynb) | J-invariant self-supervision | no | no | — (PyTorch) |
| 4 | [`4_DPS.ipynb`](4_DPS.ipynb) | Pretrained diffusion prior (generative) | no | no | `diffusers` |

All notebooks share the **same ground-truth face and the same noise** (sigma = 25/255),
so the reported PSNR is directly comparable.

### Typical results (256×256, sigma = 25/255, noisy ≈ 20.6 dB)

| BM3D | Noise2Void | Noise2Self | DPS |
|:---:|:---:|:---:|:---:|
| ~31.9 dB | ~30.6 dB | ~29.9 dB | ~30.4 dB |

(Exact numbers vary slightly with the random seed and step counts.)

## Running on Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com) → **File ▸ Upload notebook** and pick one of the `.ipynb` files (or drag it in).
2. **Runtime ▸ Change runtime type ▸ GPU** (a free T4 is plenty; only notebooks 2–4 need it).
3. **Runtime ▸ Run all.**

Each notebook downloads one CelebA-HQ face from the HuggingFace datasets server; if that
is unreachable it falls back to a bundled sample image, so the code always runs.

## Running locally

```bash
conda create -n denoising python=3.10 -y && conda activate denoising
pip install torch diffusers bm3d scikit-image matplotlib pillow requests jupyter
jupyter lab        # then open any notebook
```

## Bring your own image

Notebooks **1–3** end with a *"Try it on your own image"* section: upload any
file — colour **or** single-channel (e.g. a 16-bit microscopy TIFF), in any
value range — and it is min-max normalised to `[0,1]`, channel-handled, and
optionally downscaled before being denoised with that notebook's method. BM3D
estimates the noise level automatically (or set it by hand); Noise2Void and
Noise2Self retrain the small network on your image and need no noise level.
DPS has no such section — it requires a diffusion prior pretrained on your image
domain, which a generic upload does not provide.

## What each notebook teaches

- **BM3D** — the strong, training-free baseline. It *uses* the known noise level; the deep
  methods that follow do not.
- **Noise2Void** — train a tiny CNN on the single noisy image by hiding pixels
  ("blind spots") and predicting them from their neighbours.
- **Noise2Self** — the same blind-spot idea formalised as **J-invariance**, with a clean
  argument for why minimising self-prediction error on noisy data also denoises.
- **DPS (Diffusion Posterior Sampling)** — no training at all: borrow a diffusion model
  pretrained on CelebA-HQ (`google/ddpm-celebahq-256`) as a prior and guide its reverse
  process toward the measurement. The same loop generalises to inpainting,
  super-resolution and deblurring by changing one operator.

