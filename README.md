<h1 align="center"><sub><sup>TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows</sup></sub></h1>

<p align="center">
  <a href="https://zhenglin-cheng.com/" target="_blank">Zhenglin&nbsp;Cheng</a><sup>*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=-8XvRRIAAAAJ" target="_blank">Peng&nbsp;Sun</a><sup>*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sites.google.com/site/leeplus/" target="_blank">Jianguo&nbsp;Li</a> &ensp; <b>&middot;</b> &ensp;
  <a href="https://lins-lab.github.io/" target="_blank">Tao&nbsp;Lin</a>
</p>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project%20Page-133399.svg?logo=homepage)](https://zhenglin-cheng.com/twinflow)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-TwinFlow--Qwen--Image-yellow)](https://huggingface.co/inclusionAI/TwinFlow)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-TwinFlow--Z--Image--Turbo--exp-yellow)](https://huggingface.co/inclusionAI/TwinFlow-Z-Image-Turbo)&#160;
[![Github Repo](https://img.shields.io/badge/inclusionAI%2FTwinFlow-black?logo=github)](https://github.com/inclusionAI/TwinFlow)&#160;
<a href="https://arxiv.org/abs/2512.05150" target="_blank"><img src="https://img.shields.io/badge/Paper-b5212f.svg?logo=arxiv" height="21px"></a>
<a href="https://deepwiki.com/inclusionAI/TwinFlow"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>&#160;
[![WeChat](https://img.shields.io/badge/WeChat-green?logo=wechat&amp&color=white)](./assets/wechat.png)

</div>

**Join the WeChat Group, feel free to reach out anytime if you have any questions!👇**

<details>
<summary>👇 WeChat Group QR Code/微信群二维码 👇</summary>

<div align="center">
  <img src="./assets/wechat.png" width="1000" />
  <p style="margin-top: 8px; font-size: 14px; color: #666; font-weight: bold;">
  </p>
</div>

</details>

## 🧭 Table of Contents

- [🔥 Codebase Usage 🔥](src/README.md)
- [Inference Demo](#inference-demo)
- [Tutorials on MNIST](tutorials/README.md)

## 📰 News

- We release experimental version of faster Z-Image-Turbo!
- We release training code and better TwinFlow implementation on SD3.5 and OpenUni under `src` directory 👏🏻.
- We release tutorials on MNIST to provide core implementation of TwinFlow!
- We release **TwinFlow-Qwen-Image-v1.0**! And we are also working on **Z-Image-Turbo to make it faster!**

## ⚙️ Key Features

![](assets/twinflow_feats.jpg)

## 💪 Open-source Plans

- [x] Release inference and sampler code for TwinFlow-Qwen-Image-v1.0.
- [x] Release training tutorials on MNIST for understanding.
- [x] Release training code on SD3.5 and OpenUni.
- [x] Release faster experimental version of Z-Image-Turbo.
- [ ] Release large-scale training code.

## TwinFlow

### TwinFlow-Z-Image-Turbo-exp Visualization

<div align="center">
  <img src="assets/twinflow_z_2step.jpg" width="1000" />
  <p style="margin-top: 8px; font-size: 14px; color: #666; font-weight: bold;">
    2-NFE visualization of TwinFlow-Z-Image-Turbo-exp
  </p>
</div>

<details>
<summary>👀 Original Z-Image-Turbo 2-NFE</summary>

<div align="center">
  <img src="assets/z_turbo_2step.jpg" width="1000" />
  <p style="margin-top: 8px; font-size: 14px; color: #666; font-weight: bold;">
    2-NFE visualization of Z-Image-Turbo
  </p>
</div>

</details>

### TwinFlow-Qwen-Image Visualization

<div align="center">
  <img src="assets/demo.jpg" width="1000" />
  <p style="margin-top: 8px; font-size: 14px; color: #666; font-weight: bold;">
    2-NFE visualization of TwinFlow-Qwen-Image
  </p>
</div>

### Comparison with Qwen-Image and Qwen-Image-Lightning

<div align="center">
  <img src="assets/case1.jpg" width="1000" />
  <p style="margin-top: 16px; font-size: 14px; color: #666; font-weight: bold; max-width: 1000px;">
    Case 1: 万里长城秋景，蜿蜒盘踞于层峦叠嶂的山脉之上，砖石城墙与烽火台在暖阳下呈现古朴的土黄色，山间枫叶如火般绚烂，游客点缀其间，远山薄雾缭绕，天空湛蓝飘着几朵白云，高角度全景构图，细节丰富，光影柔和。
  </p>
</div>

---

<div align="center">
  <img src="assets/case2.jpg" width="1000" />
  <p style="margin-top: 16px; font-size: 14px; color: #666; font-weight: bold; max-width: 1000px;">
    Case2: 超高清壁纸, 梦幻光影, 少女在元宵灯会中回眸一笑, 提着一盏兔子花灯, 周围挂满明亮的灯笼, 暖色调灯光映照在脸上, 华丽的唐装, 繁复的头饰, 热闹的背景虚化, 焦外光斑美丽, 中景镜头。<br>
    Same prompt but different noise (left to right). Top to bottom shown are: Qwen-Image (50×2 NFE), TwinFlow-Qwen-Image (1-NFE), and Qwen-Image-Lightning-v2.0 (1-NFE).<br>TwinFlow-Qwen-Image generates high-quality images at 1-NFE while preserving strong diversity.
  </p>
</div>

### Overview

We introduce TwinFlow, a framework that realizes high-quality 1-step and few-step generation without the pipeline bloat.

Instead of relying on external discriminators or frozen teachers, TwinFlow creates an internal "twin trajectory". By extending the time interval to $t\in[−1,1]$, we utilize the negative time branch to map noise to "fake" data, creating a self-adversarial signal directly within the model.

Then, the model can rectify itself by minimizing the difference of the velocity fields between real trajectory and fake trajectory, i.e. the $\Delta_\mathrm{v}$. The rectification performs distribution matching as velocity matching, which gradually transforms the model into a 1-step/few-step generator.

<div align="center">
  <img src="assets/twinflow.png" alt="TwinFlow method overview" width="500" />
  <p style="margin-top: 8px; font-size: 14px; color: #666; font-weight: bold;">
    TwinFlow method overview
  </p>
</div>

Key Advantages:
- **One-model Simplicity.** We eliminate the need for any auxiliary networks. The model learns to rectify its own flow field, acting as the generator, fake/real score. No extra GPU memory is wasted on frozen teachers or discriminators during training.
- **Scalability on Large Models.** TwinFlow is **easy to scale on 20B full-parameter training** due to the one-model simplicity. In contrast, methods like VSD, SiD, and DMD/DMD2 require maintaining three separate models for distillation, which not only significantly increases memory consumption—often leading OOM, but also introduces substantial complexity when scaling to large-scale training regimes.


### Inference Demo

Install the latest diffusers:

```bash
pip install git+https://github.com/huggingface/diffusers
```

Run inference demo `inference.py`:

```python
python inference.py
```

We recommend to sample for 2~4 NFEs:

```python
# 4 NFE config
sampler_config = {
    "sampling_steps": 4,
    "stochast_ratio": 1.0,
    "extrapol_ratio": 0.0,
    "sampling_order": 1,
    "time_dist_ctrl": [1.0, 1.0, 1.0],
    "rfba_gap_steps": [0.001, 0.5],
}

# 2 NFE config
sampler_config = {
    "sampling_steps": 2,
    "stochast_ratio": 1.0,
    "extrapol_ratio": 0.0,
    "sampling_order": 1,
    "time_dist_ctrl": [1.0, 1.0, 1.0],
    "rfba_gap_steps": [0.001, 0.6],
}
```

## 📖 Citation

```bibtex
@article{cheng2025twinflow,
  title={TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows},
  author={Cheng, Zhenglin and Sun, Peng and Li, Jianguo and Lin, Tao},
  journal={arXiv preprint arXiv:2512.05150},
  year={2025}
}

@misc{sun2025anystep,
  author = {Sun, Peng and Lin, Tao},
  note   = {GitHub repository},
  title  = {Any-step Generation via N-th Order Recursive Consistent Velocity Field Estimation},
  url    = {https://github.com/LINs-lab/RCGM},
  year   = {2025}
}

@article{sun2025unified,
  title = {Unified continuous generative models},
  author = {Sun, Peng and Jiang, Yi and Lin, Tao},
  journal = {arXiv preprint arXiv:2505.07447},
  year = {2025},
  url = {https://arxiv.org/abs/2505.07447},
  archiveprefix = {arXiv},
  eprint = {2505.07447},
  primaryclass = {cs.LG}
}
```

## 🤗 Acknowledgement

TwinFlow is built upon [RCGM](https://github.com/LINs-lab/RCGM) and [UCGM](https://github.com/LINs-lab/UCGM), with much support from [InclusionAI](https://github.com/inclusionAI).

Note: The [LINs Lab](https://lins-lab.github.io/) has openings for PhD students for the Fall 2026/2027 intake. Interested candidates are encouraged to reach out.