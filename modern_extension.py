#!/usr/bin/env python3
# CS 7180 Advanced Perception
# Name: <Your Name Here>
# Date: 2025-09-28
# File: modern_extension.py
# Purpose: A *modern* low-light enhancement extension inspired by Zero-DCE (Guo et al., 2020):
#          iterative exposure correction using a per-pixel adjustment map A, but here we use a
#          simple luminance-based heuristic to compute A (no training required).
#
# Usage:
#   python modern_extension.py --input path/to/image.jpg --output out_dce.jpg --iters 8 --k1 1.0 --k2 0.0
#
# Method (Zero-DCE-inspired):
#   Iteratively apply:  I_{t+1} = I_t + A * (I_t^2 - I_t), where A in [-1,1].
#   We compute A per-pixel from luminance Y in [0,1] as: A = clip(k1*(1 - Y) + k2, -1, 1).
#   Darker pixels (low Y) receive larger positive A -> brighter results, similar to Zero-DCE behavior.
#
# Reference for inspiration:
#   Guo, C., Li, C., & Guo, J. (2020). Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE).
#   The original learns A with a CNN; we approximate A analytically from luminance for a training-free extension.
#
# Notes:
#   - This file provides a **modern-inspired extension** you can include in your report as an extra experiment.
#   - It runs on CPU and has only OpenCV + NumPy dependencies (already in requirements).
#   - You can compare results with CLAHE/SSR/MSRCR from main.py.
import os
import cv2
import argparse
import numpy as np

def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def _to_float(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr.astype(np.float32) / 255.0

def _to_uint8(img_float: np.ndarray) -> np.ndarray:
    img = np.clip(img_float * 255.0, 0, 255)
    return img.astype(np.uint8)

def compute_luminance(img_bgr_float: np.ndarray) -> np.ndarray:
    """Compute normalized luminance Y in [0,1] via standard RGB-to-Y (BT.709) coefficients."""
    # Convert BGR->RGB
    img_rgb = img_bgr_float[..., ::-1]
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    Y = 0.2126 * img_rgb[...,0] + 0.7152 * img_rgb[...,1] + 0.0722 * img_rgb[...,2]
    Y = np.clip(Y, 0.0, 1.0)
    return Y

def zerodce_lite(img_bgr: np.ndarray, k1: float = 1.0, k2: float = 0.0, iters: int = 8) -> np.ndarray:
    """Zero-DCE-inspired enhancement with an analytic A map from luminance (no training)."""
    I = _to_float(img_bgr)
    Y = compute_luminance(I)
    A = np.clip(k1 * (1.0 - Y) + k2, -1.0, 1.0)[..., None]  # shape (H,W,1), broadcast to channels
    J = I.copy()
    for _ in range(max(1, iters)):
        J = J + A * (J * J - J)
        J = np.clip(J, 0.0, 1.0)
    return _to_uint8(J)

def main():
    ap = argparse.ArgumentParser(description="Zero-DCE-inspired exposure correction (no training).")
    ap.add_argument("--input", required=True, help="Path to image (file)")
    ap.add_argument("--output", required=True, help="Path to output image")
    ap.add_argument("--k1", type=float, default=1.0, help="Strength for (1 - luminance) term")
    ap.add_argument("--k2", type=float, default=0.0, help="Bias added to A")
    ap.add_argument("--iters", type=int, default=8, help="Number of curve iterations")
    args = ap.parse_args()

    img = _read_image(args.input)
    out = zerodce_lite(img, k1=args.k1, k2=args.k2, iters=args.iters)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()
