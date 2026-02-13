<div align="center">
<p align="center"> <img src="figs/qdit4sr.png" width="540px"> </p>
</div>

# 🚀 Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer  Quantization for Real-World Image Super-Resolution

<p align="center">
  <a href="https://xunzhang1128.github.io/">Xun Zhang</a><sup>1</sup>,
  <a href="https://github.com/racoonykc">Kaicheng Yang</a><sup>1</sup>,
  <a href="https://github.com/Lukebird17">Hongliang Lu</a><sup>1</sup>,
  <a href="https://github.com/htqin">Haotong Qin</a><sup>2</sup>,
  <a href="https://github.com/guoyongcs">Yong Guo</a><sup>3</sup>,
  <a href="https://github.com/yulunzhang"><b>Yulun Zhang</b></a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Shanghai Jiao Tong University &nbsp;&nbsp;
  <sup>2</sup>ETH Zurich &nbsp;&nbsp;
  <sup>3</sup>Huawei
</p>



<div align="center">
  <a href="https://arxiv.org/abs/2602.01273" target="_blank" style="text-decoration: none;"><img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv"></a>
  <a href="https://github.com/xunzhang1128/Q-DiT4SR/releases/download/v1/Supplementary_Material.pdf"target="_blank" style="text-decoration: none;"><img src="https://img.shields.io/badge/Supplementary_material-Paper-orange.svg"></a>
  <a href="https://github.com/xun04/Q-DiT4SR/releases" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/downloads/xun04/Q-DiT4SR/total?color=green&style=flat"></a>
  <a href="https://github.com/xun04/Q-DiT4SR" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=xun04/Q-DiT4SR"></a>
  <a href="https://github.com/xun04/Q-DiT4SR/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/xun04/Q-DiT4SR?style=social"></a>
</div>

---
## 🔥🔥🔥 News

- [**2026-01-31**] Repository initial release.


### ⭐⭐⭐ If Q-DiT4SR is helpful to your projects, please help star this repo. Thanks!
---

## 📘 Abstract

> Recently, Diffusion Transformers (DiTs) have emerged in Real-World Image Super-Resolution (Real-ISR) to generate high-quality textures, yet their heavy inference burden hinders real-world deployment. While Post-Training Quantization (PTQ) is a promising solution for acceleration, existing methods in super-resolution mostly focus on U-Net architectures, whereas generic DiT quantization is typically designed for text-to-image tasks. Directly applying these methods to DiT-based super-resolution models leads to severe degradation of local textures. Therefore, we propose **Q-DiT4SR**, the first PTQ framework specifically tailored for DiT-based Real-ISR. We propose **H-SVD**, a hierarchical SVD that integrates a global low-rank branch with a local block-wise rank-1 branch under a matched parameter budget. We further propose **V**ariance-**a**ware **S**patio-**T**emporal **M**ixed Precision: **VaSMP** allocates cross-layer weight bit-widths in a data-free manner based on rate-distortion theory, while **VaTMP** schedules intra-layer activation precision across diffusion timesteps via dynamic programming (DP) with minimal calibration. Experiments on multiple real-world datasets demonstrate that our Q-DiT4SR achieves SOTA performance under both **W4A6** and **W4A4** settings. Notably, the W4A4 quantization configuration reduces model size by **5.8**$\times$ and computational operations by over **60**$\times$.

![](figs/intro1.png)

![](figs/intro2.png)

---

## 📝 Overview

![](figs/overview.png)


---


## 🔖 TODO

- [ ] Release model checkpoints.
- [ ] Release quantization and inference code.
- [ ] Release calibration set.

---

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

---

## <a name="citation"></a>📎 Citation

If you find the code helpful in your research or work, please cite the following paper.

```
@article{zhang2026qdit4sr,
  title={Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution},
  author={Zhang, Xun and Yang, Kaicheng and Lu, Hongliang and Qin, Haotong and Guo, Yong and Zhang, Yulun},
  journal={arXiv preprint arXiv:2602.01273},
  year={2026}
}
```

---

## <a name="acknowledgements"></a>💡 Acknowledgements

We thank the developers of [DiT4SR](https://github.com/Adam-duan/DiT4SR) for their open-source contributions, which have greatly facilitated our research.
