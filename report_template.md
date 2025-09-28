# Retinex-Inspired Image Enhancement (SSR/MSRCR) with CLAHE Baseline

**Author:** <Your Name>  
**Course:** CS 7180 Advanced Perception  
**Date:** 2025-09-28

## Abstract (≤200 words)
<Brief problem, method, and key findings. Mention that it's "inspired by" Retinex and compared with CLAHE.>

## 1. Introduction & Prior Work
- What image enhancement is and why it matters (low-light, contrast, color fidelity).
- Paper(s) that inspired the project (Jobson et al. 1997 MSRCR; Pizer et al. CLAHE).  
- Any other directly relevant resources consulted.

## 2. Methods
- **CLAHE baseline** in LAB (L channel). Parameters: clipLimit, tileGridSize.
- **SSR**: log(I) - log(Gσ * I); discuss sigma choice.
- **MSRCR**: multi-scale SSR + color restoration (CRF) with α, β; robust per-channel normalization.
- Optional **unsharp masking** for detail enhancement.
- Include formula snippets and short reasoning for your parameter picks.
- Provide a small diagram/flow if helpful.

## 3. Results
- Show several images (natural scenes, low-light indoor, portraits). For each: original, CLAHE, SSR, MSRCR (and optional unsharp).  
- Comment on strengths (e.g., color constancy, dynamic range) and weaknesses (e.g., haloing, noise amplification, skin tones).  
- Note any failure cases and how parameters affect outcomes.

## 4. Reflection & Acknowledgements
- What you learned about Retinex vs. histogram methods.  
- Practical tips (preprocessing, normalization).  
- Acknowledge classmates, online resources, and any LLM assistance (if used).

## References
- Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). Properties and Performance of a Center/Surround Retinex. IEEE TIP.
- Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes. IEEE TIP.
- Pizer, S. M., Amburn, E. P., Austin, J. D., et al. (1990/1994). Adaptive Histogram Equalization and its Variations.
