# Retinex-Inspired Image Enhancement (SSR/MSRCR) with CLAHE Baseline

**Course:** CS 7180 Advanced Perception  
**Author:** <Your Name>  
**Date:** 2025-09-28

This project implements an easy-to-run image enhancement pipeline based on the classic Retinex literature, with a simple CLAHE baseline for comparison. It is designed for a **solo** project and to be runnable on CPU in a few minutes.

But in image enhancement, especially **Retinex theory**, age doesn’t mean “irrelevant.” Retinex models were foundational: they explain how the human visual system maintains color constancy in different lighting. The **MSRCR (Multiscale Retinex with Color Restoration)** paper is cited thousands of times and is still considered a benchmark or baseline method even in 2025.


## Methods
- **CLAHE (baseline)**: Pizer et al., 1994. Apply CLAHE on the L channel in LAB color space.
- **SSR** (Single-Scale Retinex): Log-ratio between image and its Gaussian-smoothed version.
- **MSRCR** (Multi-Scale Retinex with Color Restoration): Jobson et al., 1997. Weighted sum of SSR across scales + color restoration; robust per-channel normalization.

Optional: **Unsharp masking** for mild sharpness boost after enhancement.

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage
Single image:
```bash
python main.py --input path/to/image.jpg --method msrcr --output out.jpg
python main.py --input path/to/image.jpg --method ssr --sigma 80 --output out_ssr.jpg
python main.py --input path/to/image.jpg --method clahe --output out_clahe.jpg
```

Folder of images (writes alongside originals with suffix):
```bash
python main.py --input path/to/folder --method msrcr --suffix _msrcr
```

Optional unsharp masking:
```bash
python main.py --input image.jpg --method msrcr --unsharp --unsharp_sigma 1.0 --unsharp_strength 0.5 --output out_sharp.jpg
```

## Tips
- MSRCR defaults (`sigmas=[15,80,250]`, equal weights) work well for many natural scenes.
- If results look washed out, reduce `beta` or use smaller sigmas (e.g., `--msr_sigmas 15 60 120`).  
- For low-light images, MSRCR often outperforms CLAHE in preserving colors; CLAHE can over-saturate or amplify noise.

## References
- Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). Properties and Performance of a Center/Surround Retinex. *IEEE Transactions on Image Processing*.
- Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes. *IEEE Transactions on Image Processing*.
- Pizer, S. M., Amburn, E. P., Austin, J. D., et al. (1990/1994). Adaptive Histogram Equalization and its Variations. *Computer Vision, Graphics, and Image Processing*, and related works.


## Extension: Modern, training-free enhancement (Zero-DCE–inspired)
We include a lightweight extension **inspired by Zero-DCE (2020)** that performs iterative exposure correction with a per-pixel adjustment map **A**, but **without any training**. We compute **A** analytically from luminance (darker pixels get stronger positive correction), then apply the standard deep-curve update:
\[ I_{t+1} = I_t + A \cdot (I_t^2 - I_t) \]
This captures the *essence* of recent deep exposure methods while remaining CPU-friendly and assignment-sized.

**Run:**
```bash
python modern_extension.py --input path/to/image.jpg --output out_dce.jpg --iters 8 --k1 1.0 --k2 0.0
```

**Compare:** Include this as an extra column in your Results section alongside CLAHE / SSR / MSRCR to earn extension points.
