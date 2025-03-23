# Testing BoxDiff for my master thesis

<div align="center">
<h1>BoxDiff 🎨 (ICCV 2023)</h1>
<h3>BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion</h3>

[Jinheng Xie](https://sierkinhane.github.io/)<sup>1</sup>&nbsp; Yuexiang Li<sup>2</sup>&nbsp; Yawen Huang<sup>2</sup>&nbsp; Haozhe Liu<sup>2,3</sup>&nbsp; Wentian Zhang<sup>2</sup> Yefeng Zheng<sup>2</sup>&nbsp; [Mike Zheng Shou](https://scholar.google.com/citations?hl=zh-CN&user=h1-3lSoAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup>

<sup>1</sup> National University of Singapore&nbsp; <sup>2</sup> Tencent Jarvis Lab&nbsp; <sup>3</sup> KAUST

[![arXiv](https://img.shields.io/badge/arXiv-<2307.10816>-<COLOR>.svg)](https://arxiv.org/abs/2307.10816)

</div>

## Quick start

```bash
conda create --name boxdiff python=3.10
conda activate boxdiff
pip install -r requirements.txt
```

## Image generation

The .csv file containing the prompts should be inside a folder named `prompts` that is posiotioned in the root of the project.

## Citation

```
@InProceedings{Xie_2023_ICCV,
    author    = {Xie, Jinheng and Li, Yuexiang and Huang, Yawen and Liu, Haozhe and Zhang, Wentian and Zheng, Yefeng and Shou, Mike Zheng},
    title     = {BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
    pages     = {7452-7461}
}
```

Acknowledgment - the code is highly based on the repository of [diffusers](https://github.com/huggingface/diffusers), [google](https://github.com/google/prompt-to-prompt), and [yuval-alaluf](https://github.com/yuval-alaluf).
