#!/usr/bin/env python3
# CS 7180 Advanced Perception
# Name: <Your Name Here>
# Date: 2025-09-28
# File: main.py
# Purpose: Retinex-inspired image enhancement (SSR/MSRCR) with CLAHE baseline and optional unsharp masking.
#
# Usage examples:
#   python main.py --input path/to/image.jpg --method msrcr --output out.jpg
#   python main.py --input path/to/folder --method clahe --suffix _clahe
#   python main.py --input path/to/image.jpg --method ssr --sigma 80 --output out_ssr.jpg
#
# Notes:
# - "Based on": Jobson, Rahman, and Woodell (1997) Retinex by Multi-Scale Retinex with Color Restoration (MSRCR).
# - "Baseline": CLAHE (Pizer et al., 1994) applied in LAB color space on L channel.
#
# Functions have short docstrings; larger functions have section comments for readability.

import os
import cv2
import argparse
import numpy as np
from typing import Sequence, Tuple

def _read_image(path: str) -> np.ndarray:
    """Read an image with OpenCV in BGR uint8 format."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def _write_image(path: str, img_bgr: np.ndarray) -> None:
    """Write an image (BGR, uint8) to disk, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, img_bgr)
    if not ok:
        raise IOError(f"Failed to write image: {path}")

def _to_float(img_bgr: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR [0,255] to float32 [0,1]."""
    return img_bgr.astype(np.float32) / 255.0

def _to_uint8(img_float: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] (or broader) to uint8 [0,255] with clipping."""
    img = np.clip(img_float * 255.0, 0, 255)
    return img.astype(np.uint8)

def apply_clahe_lab(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """CLAHE baseline in LAB color space on L channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

def unsharp_mask(img_bgr: np.ndarray, sigma: float = 1.0, strength: float = 1.0) -> np.ndarray:
    """Simple unsharp masking: out = img + strength*(img - blur(img))."""
    blur = cv2.GaussianBlur(img_bgr, (0,0), sigmaX=sigma, sigmaY=sigma)
    mask = cv2.addWeighted(img_bgr, 1.0+strength, blur, -strength, 0)
    return mask

def ssr_channel(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Single-Scale Retinex on a single channel (float32 in [0,1])."""
    # Ensure positive to avoid log(0)
    eps = 1e-6
    blur = cv2.GaussianBlur(channel, (0,0), sigmaX=sigma, sigmaY=sigma)
    retinex = np.log(channel + eps) - np.log(blur + eps)
    return retinex

def ssr(img_bgr: np.ndarray, sigma: float = 80.0, gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Single-Scale Retinex on each channel, followed by dynamic range compression to [0,1]."""
    img = _to_float(img_bgr)
    out = np.zeros_like(img)
    for c in range(3):
        out[..., c] = ssr_channel(img[..., c], sigma=sigma)
    # Normalize each channel to [0,1] robustly
    for c in range(3):
        ch = out[..., c]
        ch -= np.percentile(ch, 1)
        ch /= (np.percentile(ch, 99) - np.percentile(ch, 1) + 1e-6)
        out[..., c] = np.clip(ch, 0, 1)
    out = np.clip(gain*out + offset, 0, 1)
    return _to_uint8(out)

def msrcr(img_bgr: np.ndarray,
          sigmas: Sequence[float] = (15.0, 80.0, 250.0),
          weights: Sequence[float] = (1/3, 1/3, 1/3),
          alpha: float = 125.0,
          beta: float = 46.0,
          gain: float = 1.0,
          offset: float = 0.0) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR), simplified.
    - sigmas: Gaussian scales
    - weights: same length as sigmas; sum to 1
    - alpha, beta: parameters for color restoration (CRF)
    - gain, offset: final linear transform before clipping
    Returns BGR uint8.
    """
    img = _to_float(img_bgr)
    eps = 1e-6

    # --- Multi-Scale Retinex (MSR): weighted sum of SSR at different scales
    msr = np.zeros_like(img, dtype=np.float32)
    for w, s in zip(weights, sigmas):
        for c in range(3):
            msr[..., c] += w * ssr_channel(img[..., c], sigma=s)

    # --- Color Restoration Function (CRF)
    sum_channels = np.sum(img, axis=2, keepdims=True) + eps
    # CRF = beta * ( log(alpha * I) - log(sum_rgb) )
    crf = beta * (np.log(alpha * img + eps) - np.log(sum_channels))

    # Apply CRF
    msrcr_float = msr * crf

    # --- Simple dynamic range compression to [0,1] robustly per-channel
    out = np.zeros_like(msrcr_float)
    for c in range(3):
        ch = msrcr_float[..., c]
        ch -= np.percentile(ch, 1)
        ch /= (np.percentile(ch, 99) - np.percentile(ch, 1) + 1e-6)
        out[..., c] = np.clip(ch, 0, 1)

    out = np.clip(gain*out + offset, 0, 1)
    return _to_uint8(out)

def process_path(input_path: str,
                 method: str,
                 output: str = None,
                 suffix: str = "_enh",
                 ssr_sigma: float = 80.0,
                 msr_sigmas: Sequence[float] = (15.0, 80.0, 250.0),
                 msr_weights: Sequence[float] = (1/3, 1/3, 1/3),
                 alpha: float = 125.0,
                 beta: float = 46.0,
                 clahe_clip: float = 2.0,
                 clahe_grid: int = 8,
                 unsharp: bool = False,
                 unsharp_sigma: float = 1.0,
                 unsharp_strength: float = 0.5) -> None:
    """
    Process an image or all images in a folder with the chosen method: 'clahe', 'ssr', or 'msrcr'.
    Optionally apply unsharp masking after enhancement.
    """
    def enhance(img_bgr: np.ndarray) -> np.ndarray:
        if method == "clahe":
            out = apply_clahe_lab(img_bgr, clip_limit=clahe_clip, tile_grid_size=clahe_grid)
        elif method == "ssr":
            out = ssr(img_bgr, sigma=ssr_sigma)
        elif method == "msrcr":
            out = msrcr(img_bgr, sigmas=msr_sigmas, weights=msr_weights, alpha=alpha, beta=beta)
        else:
            raise ValueError("Unknown method: choose from 'clahe', 'ssr', 'msrcr'")
        if unsharp:
            out = unsharp_mask(out, sigma=unsharp_sigma, strength=unsharp_strength)
        return out

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_ext:
                in_f = os.path.join(input_path, fname)
                img = _read_image(in_f)
                out = enhance(img)
                out_name = os.path.splitext(fname)[0] + suffix + ext
                out_f = os.path.join(input_path, out_name) if output is None else os.path.join(output, out_name)
                _write_image(out_f, out)
        print("Done.")
    else:
        img = _read_image(input_path)
        out = enhance(img)
        if output is None:
            root, ext = os.path.splitext(input_path)
            out_f = root + suffix + ext
        else:
            out_f = output
        _write_image(out_f, out)
        print(f"Wrote: {out_f}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retinex-inspired enhancement (SSR/MSRCR) + CLAHE baseline")
    p.add_argument("--input", required=True, help="Path to image file or folder")
    p.add_argument("--method", required=True, choices=["clahe", "ssr", "msrcr"], help="Enhancement method")
    p.add_argument("--output", default=None, help="Output file or folder (if input is folder, treated as folder)")
    p.add_argument("--suffix", default="_enh", help="Suffix for output filenames if output not provided")
    # SSR
    p.add_argument("--sigma", type=float, default=80.0, help="Gaussian sigma for SSR")
    # MSRCR
    p.add_argument("--msr_sigmas", type=float, nargs="+", default=[15.0, 80.0, 250.0], help="Sigmas for MSRCR")
    p.add_argument("--msr_weights", type=float, nargs="+", default=[1/3, 1/3, 1/3], help="Weights for MSRCR (same length as sigmas)")
    p.add_argument("--alpha", type=float, default=125.0, help="MSRCR color restoration alpha")
    p.add_argument("--beta", type=float, default=46.0, help="MSRCR color restoration beta")
    # CLAHE
    p.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clip limit")
    p.add_argument("--clahe_grid", type=int, default=8, help="CLAHE tile grid size")
    # Optional unsharp
    p.add_argument("--unsharp", action="store_true", help="Apply unsharp masking after enhancement")
    p.add_argument("--unsharp_sigma", type=float, default=1.0, help="Gaussian sigma for unsharp mask")
    p.add_argument("--unsharp_strength", type=float, default=0.5, help="Unsharp strength")
    return p

def main():
    args = build_argparser().parse_args()
    # Harmonize weights length
    if len(args.msr_sigmas) != len(args.msr_weights):
        raise ValueError("msr_sigmas and msr_weights must have the same length.")
    wsum = sum(args.msr_weights)
    if abs(wsum - 1.0) > 1e-6:
        args.msr_weights = [w / wsum for w in args.msr_weights]

    process_path(
        input_path=args.input,
        method=args.method,
        output=args.output,
        suffix=args.suffix,
        ssr_sigma=args.sigma,
        msr_sigmas=args.msr_sigmas,
        msr_weights=args.msr_weights,
        alpha=args.alpha,
        beta=args.beta,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        unsharp=args.unsharp,
        unsharp_sigma=args.unsharp_sigma,
        unsharp_strength=args.unsharp_strength,
    )

if __name__ == "__main__":
    main()
