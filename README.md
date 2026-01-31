# Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer  Quantization for Real-World Image Super-Resolution
[Xun Zhang](https://xun04.github.io/), [Kaicheng Yang](https://github.com/racoonykc), [Hongliang Lu](https://github.com/Lukebird17), [Haotong Qin](https://github.com/htqin), [Yong Guo](https://github.com/guoyongcs) and [Yulun Zhang](https://github.com/yulunzhang).

<div>
<a href="https://github.com/xun04/Q-DiT4SR/releases" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/downloads/xun04/Q-DiT4SR/total?color=green&style=flat"></a>
<a href="https://github.com/xun04/Q-DiT4SR" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=xun04/Q-DiT4SR"></a>
<a href="https://github.com/xun04/Q-DiT4SR/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/xun04/Q-DiT4SR?style=social"></a>
</div>

#### 🔥🔥🔥 News

- **2026-01-31:** Repository initial release.

---

### Abstract

> Recently, Diffusion Transformers (DiTs) have emerged in Real-World Image Super-Resolution (Real-ISR) to generate high-quality textures, yet their heavy inference burden hinders real-world deployment. While Post-Training Quantization (PTQ) is a promising solution for acceleration, existing methods in super-resolution mostly focus on U-Net architectures, whereas generic DiT quantization is typically designed for text-to-image tasks. Directly applying these methods to DiT-based super-resolution models leads to severe degradation of local textures. Therefore, we propose **Q-DiT4SR**, the first PTQ framework specifically tailored for DiT-based Real-ISR. We propose **H-SVD**, a hierarchical SVD that integrates a global low-rank branch with a local block-wise rank-1 branch under a matched parameter budget. We further propose **V**ariance-**a**ware **S**patio-**T**emporal **M**ixed Precision: **VaSMP** allocates cross-layer weight bit-widths in a data-free manner based on rate-distortion theory, while **VaTMP** schedules intra-layer activation precision across diffusion timesteps via dynamic programming (DP) with minimal calibration. Experiments on multiple real-world datasets demonstrate that our Q-DiT4SR achieves SOTA performance under both **W4A6** and **W4A4** settings. Notably, the W4A4 quantization configuration reduces model size by **5.8**$\times$ and computational operations by over **60**$\times$. We will release our code and models to facilitate reproducibility and future research.

![](figs/intro1.png)

![](figs/intro2.png)

---

### Overview

![](figs/overview.png)


---


## 🔖 TODO

- [ ] Release model checkpoints.
- [ ] Release inference code.
- [ ] Release quantization code.
- [ ] Release calibration set.


### Visual Comparison

![](figs/comparison.png)


## <a name="results"></a>🔎 Results

We achieve SOTA Real-ISR performance on both W4A6 and W4A4 settings based on DiT backbones.

<details open>
<summary>Quantitative Results (click to expand)</summary>

- Results in Tab. 1 of the main paper

<p align="center">
  <img width="900" src="figs/quantitative.png">
</p>
</details>

<details open>
<summary>Qualitative Results (click to expand)</summary>

- Results in Fig. 6 of the main paper

<p align="center">
  <img width="900" src="figs/comparison.png">
</p>
</details>


