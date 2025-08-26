# Awesome-Visual-Tokenizers [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) <!-- omit in toc -->

This is a repository for organizing papers, codes and other resources related to visual tokenizers.

#### :thinking: What are visual tokenizers?

A visual tokenizer is a mechanism that maps input visual signals (such as images or videos) into a set of compact and structured visual units (tokens), which may be continuous vectors, discrete indices, or a hybrid of both. A core requirement of a visual tokenizer is that the generated visual units must possess sufficient representational capacity to enable high-quality reconstruction of the original visual input through a corresponding decoder or generator.

**Note**: This definition is not intended to represent a universally accepted or formally established definition in the academic community. Readers and users should be aware that different works in the literature may adopt varying definitions or criteria for what constitutes a visual tokenizer.

#### :high_brightness: This project is still on-going, pull requests are welcomed!!

If you have any suggestions (missing papers, new papers, or typos), please feel free to edit and pull a request. Just letting us know the title of papers can also be a great contribution to us. You can do this by open issue or contact us directly via email.

#### :star: If you find this repo useful, please star it!!!

## Table of Contents <!-- omit in toc -->

- [Continuous](#continuous)
- [Discrete](#discrete)
- [Hybrid](#hybrid)

## Visual Tokenizers

### Continuous

+ [NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale](https://arxiv.org/pdf/2508.10711) (Aug 14, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.10711)
  [![Star](https://img.shields.io/github/stars/stepfun-ai/NextStep-1.svg?style=social&label=Star)](https://github.com/stepfun-ai/NextStep-1)

+ [Qwen-Image Technical Report](https://arxiv.org/pdf/2508.02324) (Aug 4, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.02324)
  [![Star](https://img.shields.io/github/stars/QwenLM/Qwen-Image.svg?style=social&label=Star)](https://github.com/QwenLM/Qwen-Image)

+ [DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space](https://arxiv.org/pdf/2508.00413) (Aug 1, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.00413)
  [![Star](https://img.shields.io/github/stars/dc-ai-projects/DC-Gen.svg?style=social&label=Star)](https://github.com/dc-ai-projects/DC-Gen)

+ [Latent Denoising Makes Good Visual Tokenizers](https://arxiv.org/pdf/2507.15856) (Jul 21, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.15856)
  [![Star](https://img.shields.io/github/stars/Jiawei-Yang/DeTok.svg?style=social&label=Star)](https://github.com/Jiawei-Yang/DeTok)

+ [Seedance 1.0: Exploring the Boundaries of Video Generation Models](https://arxiv.org/pdf/2506.09113) (Jun 10, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.09113)

+ [VIVAT: Virtuous Improving VAE Training through Artifact Mitigation](https://arxiv.org/pdf/2506.07863) (Jun 9, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.07863)

+ [MAGI-1: Autoregressive Video Generation at Scale](https://arxiv.org/pdf/2505.13211) (May 19, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.13211)
  [![Star](https://img.shields.io/github/stars/SandAI-org/MAGI-1.svg?style=social&label=Star)](https://github.com/SandAI-org/MAGI-1)

+ [BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset](https://arxiv.org/pdf/2505.09568) (May 14, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.09568)
  [![Star](https://img.shields.io/github/stars/JiuhaiChen/BLIP3o.svg?style=social&label=Star)](https://github.com/JiuhaiChen/BLIP3o)

+ [REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](https://arxiv.org/pdf/2504.10483) (Apr 14, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.10483)
  [![Star](https://img.shields.io/github/stars/End2End-Diffusion/REPA-E.svg?style=social&label=Star)](https://github.com/End2End-Diffusion/REPA-E)

+ [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/pdf/2503.20314) (Mar 26, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.20314)
  [![Star](https://img.shields.io/github/stars/Wan-Video/Wan2.1.svg?style=social&label=Star)](https://github.com/Wan-Video/Wan2.1)

+ [TULIP: Towards Unified Language-Image Pretraining](https://arxiv.org/pdf/2503.15485) (Mar 19, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.15485)
  [![Star](https://img.shields.io/github/stars/tulip-berkeley/open_clip.svg?style=social&label=Star)](https://github.com/tulip-berkeley/open_clip)

+ [LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](https://arxiv.org/pdf/2503.14325) (Mar 18, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.14325)
  [![Star](https://img.shields.io/github/stars/westlake-repl/LeanVAE.svg?style=social&label=Star)](https://github.com/westlake-repl/LeanVAE)

+ [FlowTok: Flowing Seamlessly Across Text and Image Tokens](https://arxiv.org/pdf/2503.10772) (Mar 13, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.10772)
  [![Star](https://img.shields.io/github/stars/bytedance/1d-tokenizer.svg?style=social&label=Star)](https://github.com/bytedance/1d-tokenizer)

+ [Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k](https://arxiv.org/pdf/2503.09642) (Mar 12, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.09642)
  [![Star](https://img.shields.io/github/stars/hpcaitech/Open-Sora.svg?style=social&label=Star)](https://github.com/hpcaitech/Open-Sora)

+ [Improving the Diffusability of Autoencoders](https://arxiv.org/pdf/2502.14831) (Feb 20, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.14831)
  [![Star](https://img.shields.io/github/stars/snap-research/diffusability.svg?style=social&label=Star)](https://github.com/snap-research/diffusability)

+ [EQ-VAE: Equivariance Regularized Latent Space for Improved Generative Image Modeling](https://arxiv.org/pdf/2502.09509) (Feb 13, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.09509)
  [![Star](https://img.shields.io/github/stars/zelaki/eqvae.svg?style=social&label=Star)](https://github.com/zelaki/eqvae)

+ [Masked Autoencoders Are Effective Tokenizers for Diffusion Models](https://arxiv.org/pdf/2502.03444) (Feb 5, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.03444)
  [![Star](https://img.shields.io/github/stars/Hhhhhhao/continuous_tokenizer.svg?style=social&label=Star)](https://github.com/Hhhhhhao/continuous_tokenizer)

+ [Diffusion Autoencoders are Scalable Image Tokenizers](https://arxiv.org/pdf/2501.18593) (Jan 30, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.18593)
  [![Star](https://img.shields.io/github/stars/yinboc/dito.svg?style=social&label=Star)](https://github.com/yinboc/dito)

+ [CAT: Content-Adaptive Image Tokenization](https://arxiv.org/pdf/2501.03120) (Jan 6, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.03120)

+ [Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models](https://arxiv.org/pdf/2501.01423) (Jan 2, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.01423)
  [![Star](https://img.shields.io/github/stars/hustvl/LightningDiT.svg?style=social&label=Star)](https://github.com/hustvl/LightningDiT)

+ [LTX-Video: Realtime Video Latent Diffusion](https://arxiv.org/pdf/2501.00103) (Dec, 30, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.00103)
  [![Star](https://img.shields.io/github/stars/Lightricks/LTX-Video.svg?style=social&label=Star)](https://github.com/Lightricks/LTX-Video)

+ [Open-Sora: Democratizing Efficient Video Production for All](https://arxiv.org/pdf/2412.20404) (Dec 29, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.20404)
  [![Star](https://img.shields.io/github/stars/hpcaitech/Open-Sora.svg?style=social&label=Star)](https://github.com/hpcaitech/Open-Sora)

+ [Large Motion Video Autoencoding with Cross-modal Video VAE](https://arxiv.org/pdf/2412.17805) (Dec 23, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.17805)
  [![Star](https://img.shields.io/github/stars/VideoVerses/VideoVAEPlus.svg?style=social&label=Star)](https://github.com/VideoVerses/VideoVAEPlus)

+ [SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer](https://arxiv.org/pdf/2412.10958) (Dec 14, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.10958)
  [![Star](https://img.shields.io/github/stars/Hhhhhhao/continuous_tokenizer.svg?style=social&label=Star)](https://github.com/Hhhhhhao/continuous_tokenizer)

+ [Multimodal Latent Language Modeling with Next-Token Diffusion](https://arxiv.org/pdf/2412.08635) (Dec 11, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.08635)
  [![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)](https://github.com/microsoft/unilm)

+ [HunyuanVideo: A Systematic Framework For Large Video Generative Models](https://arxiv.org/pdf/2412.03603) (Dec 3, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.03603)
  [![Star](https://img.shields.io/github/stars/Tencent-Hunyuan/HunyuanVideo.svg?style=social&label=Star)](https://github.com/Tencent-Hunyuan/HunyuanVideo)

+ [Open-Sora Plan: Open-Source Large Video Generation Model](https://arxiv.org/pdf/2412.00131) (Nov 28, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.00131)
  [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social&label=Star)](https://github.com/PKU-YuanGroup/Open-Sora-Plan)

+ [WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model](https://arxiv.org/abs/2411.17459) (Nov 26, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.17459)
  [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/WF-VAE.svg?style=social&label=Star)](https://github.com/PKU-YuanGroup/WF-VAE)

+ [REDUCIO! Generating 1024 $\times$ 1024 Video within 16 Seconds using Extremely Compressed Motion Latents](https://arxiv.org/pdf/2411.13552) (Nov 20, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.13552)
  [![Star](https://img.shields.io/github/stars/microsoft/Reducio-VAE.svg?style=social&label=Star)](https://github.com/microsoft/Reducio-VAE)

+ [Improved Video VAE for Latent Video Diffusion Model](https://arxiv.org/pdf/2411.06449) (Nov 10, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.06449)
  [![Star](https://img.shields.io/github/stars/ali-vilab/iv-vae.svg?style=social&label=Star)](https://github.com/ali-vilab/iv-vae)

+ [Allegro: Open the Black Box of Commercial-Level Video Generation Model](https://arxiv.org/pdf/2410.15458) (Oct 20, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.15458)
  [![Star](https://img.shields.io/github/stars/rhymes-ai/Allegro.svg?style=social&label=Star)](https://github.com/rhymes-ai/Allegro)

+ [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://arxiv.org/pdf/2410.10733) (Oct 14, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.10733)
  [![Star](https://img.shields.io/github/stars/mit-han-lab/efficientvit.svg?style=social&label=Star)](https://github.com/mit-han-lab/efficientvit)

+ [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/pdf/2410.10629) (Oct 14, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.10629)
  [![Star](https://img.shields.io/github/stars/NVlabs/Sana.svg?style=social&label=Star)](https://github.com/NVlabs/Sana)

+ [Epsilon-VAE: Denoising as Visual Decoding](https://arxiv.org/pdf/2410.04081) (Oct 5, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.04081)

+ [OD-VAE: An Omni-dimensional Video Compressor for Improving Latent Video Diffusion Model](https://arxiv.org/pdf/2409.01199) (Sep 2, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.01199)
  [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social&label=Star)](https://github.com/PKU-YuanGroup/Open-Sora-Plan)

+ [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/pdf/2408.11039) (Aug 20, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.11039)
  [![Star](https://img.shields.io/github/stars/lucidrains/transfusion-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/transfusion-pytorch)

+ [CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/pdf/2408.06072) (Aug 12, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.06072)
  [![Star](https://img.shields.io/github/stars/THUDM/CogVideo.svg?style=social&label=Star)](https://github.com/THUDM/CogVideo)

+ [FLUX](https://bfl.ai/announcements/24-08-01-bfl) (Aug 1, 2024. BFL)
  [![Blog](https://img.shields.io/badge/Blog-b31b1b.svg)](https://bfl.ai/announcements/24-08-01-bfl)
  [![Star](https://img.shields.io/github/stars/black-forest-labs/flux.svg?style=social&label=Star)](https://github.com/black-forest-labs/flux)

+ [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/pdf/2406.11838) (Jun 17, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.11838)
  [![Star](https://img.shields.io/github/stars/LTH14/mar.svg?style=social&label=Star)](https://github.com/LTH14/mar)

+ [CV-VAE: A Compatible Video VAE for Latent Generative Video Models](https://arxiv.org/pdf/2405.20279) (May 30, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.20279)
  [![Star](https://img.shields.io/github/stars/AILab-CVC/CV-VAE.svg?style=social&label=Star)](https://github.com/AILab-CVC/CV-VAE)

+ [EasyAnimate: A High-Performance Long Video Generation Method based on Transformer Architecture](https://arxiv.org/pdf/2405.18991) (May 29, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.18991)
  [![Star](https://img.shields.io/github/stars/aigc-apps/EasyAnimate.svg?style=social&label=Star)](https://github.com/aigc-apps/EasyAnimate)

+ [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206) (Mar 5, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.03206)
  [![Star](https://img.shields.io/github/stars/Stability-AI/sd3.5.svg?style=social&label=Star)](https://github.com/Stability-AI/sd3.5)

+ [SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation](https://arxiv.org/pdf/2404.14396) (Apr 22, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.14396)
  [![Star](https://img.shields.io/github/stars/AILab-CVC/SEED-X.svg?style=social&label=Star)](https://github.com/AILab-CVC/SEED-X)

+ [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952) (Jul 4, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.01952)
  [![Star](https://img.shields.io/github/stars/Stability-AI/generative-models.svg?style=social&label=Star)](https://github.com/Stability-AI/generative-models)

+ [Diffusion Models as Masked Autoencoders](https://arxiv.org/pdf/2304.03283) (Apr 6, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.03283)

+ [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752) (Dec 20, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.10752)
  [![Star](https://img.shields.io/github/stars/CompVis/latent-diffusion.svg?style=social&label=Star)](https://github.com/CompVis/latent-diffusion)

+ [Diffusion Autoencoders: Toward a Meaningful and Decodable Representation](https://arxiv.org/pdf/2111.15640) (Nov 30, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.15640)
  [![Star](https://img.shields.io/github/stars/phizaz/diffae.svg?style=social&label=Star)](https://github.com/phizaz/diffae)

+ [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377) (Nov 11, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.06377)
  [![Star](https://img.shields.io/github/stars/facebookresearch/mae.svg?style=social&label=Star)](https://github.com/facebookresearch/mae)

+ [Simple and Effective VAE Training with Calibrated Decoders](https://arxiv.org/pdf/2006.13202) (Jun 23, 2020. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2006.13202)
  [![Star](https://img.shields.io/github/stars/orybkin/sigma-vae-pytorch.svg?style=social&label=Star)](https://github.com/orybkin/sigma-vae-pytorch)

+ [$\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl) (Feb 6, 2017. OpenReview)
  [![OpenReview](https://img.shields.io/badge/OpenReview-b31b1b.svg)](https://openreview.net/forum?id=Sy2fzU9gl)
  [![Star](https://img.shields.io/github/stars/1Konny/Beta-VAE.svg?style=social&label=Star)](https://github.com/1Konny/Beta-VAE)

+ [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114) (Dec 20, 2013. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1312.6114)
  [![Star](https://img.shields.io/github/stars/AntixK/PyTorch-VAE.svg?style=social&label=Star)](https://github.com/AntixK/PyTorch-VAE)

### Discrete

+ [X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again](https://arxiv.org/pdf/2507.22058) (Jul 29, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.22058)
  [![Star](https://img.shields.io/github/stars/X-Omni-Team/X-Omni.svg?style=social&label=Star)](https://github.com/X-Omni-Team/X-Omni)

+ [Quantize-then-Rectify: Efficient VQ-VAE Training](https://arxiv.org/pdf/2507.10547) (Jul 14, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.10547)
  [![Star](https://img.shields.io/github/stars/Neur-IO/ReVQ.svg?style=social&label=Star)](https://github.com/Neur-IO/ReVQ)

+ [MGVQ: Could VQ-VAE Beat VAE? A Generalizable Tokenizer with Multi-group Quantization](https://arxiv.org/pdf/2507.07997) (Jul 10, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.07997)
  [![Star](https://img.shields.io/github/stars/MKJia/MGVQ.svg?style=social&label=Star)](https://github.com/MKJia/MGVQ)

+ [Hita: Holistic Tokenizer for Autoregressive Image Generation](https://arxiv.org/pdf/2507.02358) (Jul 3, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.02358)
  [![Star](https://img.shields.io/github/stars/CVMI-Lab/Hita.svg?style=social&label=Star)](https://github.com/CVMI-Lab/Hita)

+ [AliTok: Towards Sequence Modeling Alignment between Tokenizer and Autoregressive Model](https://arxiv.org/pdf/2506.05289) (Jun 5, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.05289)
  [![Star](https://img.shields.io/github/stars/ali-vilab/alitok.svg?style=social&label=Star)](https://github.com/ali-vilab/alitok)

+ [Images are Worth Variable Length of Representations](https://arxiv.org/pdf/2506.03643) (Jun 4, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.03643)
  [![Star](https://img.shields.io/github/stars/mao1207/DOVE.svg?style=social&label=Star)](https://github.com/mao1207/DOVE)

+ [Selftok: Discrete Visual Tokens of Autoregression, by Diffusion, and for Reasoning](https://arxiv.org/pdf/2505.07538) (May 12, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.07538)
  [![Star](https://img.shields.io/github/stars/selftok-team/SelftokTokenizer.svg?style=social&label=Star)](https://github.com/selftok-team/SelftokTokenizer)

+ [TVC: Tokenized Video Compression with Ultra-Low Bitrate](https://arxiv.org/pdf/2504.16953) (Apr 22, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.16953)

+ [Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens](https://arxiv.org/pdf/2504.14666) (Apr 20, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.14666)
  [![Star](https://img.shields.io/github/stars/selftok-team/SelftokTokenizer.svg?style=social&label=Star)](https://github.com/selftok-team/SelftokTokenizer)

+ [GigaTok: Scaling Visual Tokenizers to 3 Billion Parameters for Autoregressive Image Generation](https://arxiv.org/pdf/2504.08736) (Apr 11, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.08736)
  [![Star](https://img.shields.io/github/stars/SilentView/GigaTok.svg?style=social&label=Star)](https://github.com/SilentView/GigaTok)

+ [VARGPT-v1.1: Improve Visual Autoregressive Large Unified Model via Iterative Instruction Tuning and Reinforcement Learning](https://arxiv.org/pdf/2504.02949) (Apr 3, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.02949)
  [![Star](https://img.shields.io/github/stars/VARGPT-family/VARGPT-v1.1.svg?style=social&label=Star)](https://github.com/VARGPT-family/VARGPT-v1.1)

+ [MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization](https://arxiv.org/pdf/2504.00999) (Apr 1, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.00999)
  [![Star](https://img.shields.io/github/stars/ApexGen-X/MergeVQ.svg?style=social&label=Star)](https://github.com/ApexGen-X/MergeVQ)

+ [CODA: Repurposing Continuous VAEs for Discrete Tokenization](https://arxiv.org/pdf/2503.17760) (Mar 22, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.17760)
  [![Star](https://img.shields.io/github/stars/LeapLabTHU/CODA.svg?style=social&label=Star)](https://github.com/LeapLabTHU/CODA)

+ [Flow to the Mode: Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](https://arxiv.org/pdf/2503.11056) (Mar 14, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.11056)
  [![Star](https://img.shields.io/github/stars/kylesargent/FlowMo.svg?style=social&label=Star)](https://github.com/kylesargent/FlowMo)

+ [V2Flow: Unifying Visual Tokenization and Large Language Model Vocabularies for Autoregressive Image Generation](https://arxiv.org/pdf/2503.07493) (Mar 10, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.07493)
  [![Star](https://img.shields.io/github/stars/zhangguiwei610/V2Flow.svg?style=social&label=Star)](https://github.com/zhangguiwei610/V2Flow)

+ [UniTok: A Unified Tokenizer for Visual Generation and Understanding](https://arxiv.org/abs/2502.20321) (Feb 27, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.20321)
  [![Star](https://img.shields.io/github/stars/FoundationVision/UniTok.svg?style=social&label=Star)](https://github.com/FoundationVision/UniTok)

+ [FlexTok: Resampling Images into 1D Token Sequences of Flexible Length](https://arxiv.org/pdf/2502.13967) (Feb 19, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.13967)
  [![Star](https://img.shields.io/github/stars/apple/ml-flextok.svg?style=social&label=Star)](https://github.com/apple/ml-flextok)

+ [QLIP: Text-Aligned Visual Tokenization Unifies Auto-Regressive Multimodal Understanding and Generation](https://arxiv.org/pdf/2502.05178) (Feb 7, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.05178)
  [![Star](https://img.shields.io/github/stars/NVlabs/QLIP.svg?style=social&label=Star)](https://github.com/NVlabs/QLIP)

+ [VARGPT: Unified Understanding and Generation in a Visual Autoregressive Multimodal Large Language Model](https://arxiv.org/pdf/2501.12327) (Jan 21, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.12327)
  [![Star](https://img.shields.io/github/stars/VARGPT-family/VARGPT.svg?style=social&label=Star)](https://github.com/VARGPT-family/VARGPT)

+ [One-D-Piece: Image Tokenizer Meets Quality-Controllable Compression](https://arxiv.org/pdf/2501.10064) (Jan 17, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.10064)
  [![Star](https://img.shields.io/github/stars/turingmotors/One-D-Piece.svg?style=social&label=Star)](https://github.com/turingmotors/One-D-Piece)

+ [Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](https://arxiv.org/pdf/2412.10208) (Dec 13, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.10208)

+ [SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization](https://arxiv.org/pdf/2412.10443) (Dec 11, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.10443)

+ [ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance](https://arxiv.org/pdf/2412.06673) (Dec 9, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.06673)

+ [TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](https://arxiv.org/pdf/2412.03069) (Dec 4, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.03069)
  [![Star](https://img.shields.io/github/stars/ByteFlow-AI/TokenFlow.svg?style=social&label=Star)](https://github.com/ByteFlow-AI/TokenFlow)

+ [Scalable Image Tokenization with Index Backpropagation Quantization](https://arxiv.org/pdf/2412.02692) (Dec 3, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.02692)
  [![Star](https://img.shields.io/github/stars/TencentARC/SEED-Voken.svg?style=social&label=Star)](https://github.com/TencentARC/SEED-Voken)

+ [Factorized Visual Tokenization and Generation](https://arxiv.org/pdf/2411.16681) (Nov 25, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.16681)
  [![Star](https://img.shields.io/github/stars/showlab/FQGAN.svg?style=social&label=Star)](https://github.com/showlab/FQGAN)

+ [Image Understanding Makes for A Good Tokenizer for Image Generation](https://arxiv.org/pdf/2411.04406) (Nov 7, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.04406)
  [![Star](https://img.shields.io/github/stars/magic-research/vector_quantization.svg?style=social&label=Star)](https://github.com/magic-research/vector_quantization)

+ [Adaptive Length Image Tokenization via Recurrent Allocation](https://arxiv.org/pdf/2411.02393) (Nov 4, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.02393)
  [![Star](https://img.shields.io/github/stars/ShivamDuggal4/adaptive-length-tokenizer.svg?style=social&label=Star)](https://github.com/ShivamDuggal4/adaptive-length-tokenizer)

+ [Addressing Representation Collapse in Vector Quantized Models with One Linear Layer](https://arxiv.org/pdf/2411.02038) (Nov 4, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.02038)
  [![Star](https://img.shields.io/github/stars/youngsheen/SimVQ.svg?style=social&label=Star)](https://github.com/youngsheen/SimVQ)

+ [Randomized Autoregressive Visual Generation](https://arxiv.org/pdf/2411.00776) (Nov 1, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.00776)
  [![Star](https://img.shields.io/github/stars/bytedance/1d-tokenizer.svg?style=social&label=Star)](https://github.com/bytedance/1d-tokenizer)

+ [LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior](https://arxiv.org/pdf/2410.21264) (Oct 28, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.21264)
  [![Star](https://img.shields.io/github/stars/hywang66/LARP.svg?style=social&label=Star)](https://github.com/hywang66/LARP)

+ [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/pdf/2409.18869) (Sep 27, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.18869)
  [![Star](https://img.shields.io/github/stars/baaivision/Emu3.svg?style=social&label=Star)](https://github.com/baaivision/Emu3)

+ [MaskBit: Embedding-free Image Generation via Bit Tokens](https://arxiv.org/pdf/2409.16211) (Sep 24, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.16211)
  [![Star](https://img.shields.io/github/stars/markweberdev/maskbit.svg?style=social&label=Star)](https://github.com/markweberdev/maskbit)

+ [VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation](https://arxiv.org/pdf/2409.04429) (Sep 6, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.04429)
  [![Star](https://img.shields.io/github/stars/mit-han-lab/vila-u.svg?style=social&label=Star)](https://github.com/mit-han-lab/vila-u)

+ [Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation](https://arxiv.org/pdf/2409.04410) (Sep 6, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.04410)
  [![Star](https://img.shields.io/github/stars/TencentARC/SEED-Voken.svg?style=social&label=Star)](https://github.com/TencentARC/SEED-Voken)

+ [Show-o: One Single Transformer to Unify Multimodal Understanding and Generation](https://arxiv.org/pdf/2408.12528) (Aug 22, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.12528)
  [![Star](https://img.shields.io/github/stars/showlab/Show-o.svg?style=social&label=Star)](https://github.com/showlab/Show-o)

+ [Scaling the Codebook Size of VQGAN to 100,000 with a Utilization Rate of 99%](https://arxiv.org/pdf/2406.11837) (Jun 17, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.11837)
  [![Star](https://img.shields.io/github/stars/zh460045050/VQGAN-LC.svg?style=social&label=Star)](https://github.com/zh460045050/VQGAN-LC)

+ [An Image is Worth 32 Tokens for Reconstruction and Generation](https://arxiv.org/pdf/2406.07550) (Jun 11, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.07550)
  [![Star](https://img.shields.io/github/stars/bytedance/1d-tokenizer.svg?style=social&label=Star)](https://github.com/bytedance/1d-tokenizer)

+ [Image and Video Tokenization with Binary Spherical Quantization](https://arxiv.org/pdf/2406.07548) (Jun 11, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.07548)
  [![Star](https://img.shields.io/github/stars/zhaoyue-zephyrus/bsq-vit.svg?style=social&label=Star)](https://github.com/zhaoyue-zephyrus/bsq-vit)

+ [Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/pdf/2406.06525) (Jun 10, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.06525)
  [![Star](https://img.shields.io/github/stars/FoundationVision/LlamaGen.svg?style=social&label=Star)](https://github.com/FoundationVision/LlamaGen)

+ [LG-VQ: Language-Guided Codebook Learning](https://arxiv.org/pdf/2405.14206) (May 23, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.14206)
  [![Star](https://img.shields.io/github/stars/GuotaoLiang/LG-VQ.svg?style=social&label=Star)](https://github.com/GuotaoLiang/LG-VQ-language-guided-codebook-learning)

+ [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/pdf/2405.09818) (May 16, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.09818)
  [![Star](https://img.shields.io/github/stars/facebookresearch/chameleon.svg?style=social&label=Star)](https://github.com/facebookresearch/chameleon)

+ [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/pdf/2404.02905) (Apr 3, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.02905)
  [![Star](https://img.shields.io/github/stars/FoundationVision/VAR.svg?style=social&label=Star)](https://github.com/FoundationVision/VAR)

+ [HQ-VAE: Hierarchical Discrete Representation Learning with Variational Bayes](https://arxiv.org/pdf/2401.00365) (Dec 31, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.00365)

+ [Sequential Modeling Enables Scalable Learning for Large Vision Models](https://arxiv.org/pdf/2312.00785) (Dec 1, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.00785)
  [![Star](https://img.shields.io/github/stars/ytongbai/LVM.svg?style=social&label=Star)](https://github.com/ytongbai/LVM)

+ [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/pdf/2310.05737) (Oct 9, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.05737)
  [![Star](https://img.shields.io/github/stars/lucidrains/magvit2-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/magvit2-pytorch)

+ [Efficient-VQGAN: Towards High-Resolution Image Generation with Efficient Vision Transformers](https://arxiv.org/pdf/2310.05400) (Oct 9, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.05400)

+ [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/pdf/2309.15505) (Sep 27, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.15505)
  [![Star](https://img.shields.io/github/stars/lucidrains/vector-quantize-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/vector-quantize-pytorch)

+ [Online Clustered Codebook](https://arxiv.org/pdf/2307.15139) (Jul 27, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.15139)
  [![Star](https://img.shields.io/github/stars/lyndonzheng/CVQ-VAE.svg?style=social&label=Star)](https://github.com/lyndonzheng/CVQ-VAE)

+ [Planting a SEED of Vision in Large Language Model](https://arxiv.org/pdf/2307.08041) (Jul 16, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.08041)
  [![Star](https://img.shields.io/github/stars/AILab-CVC/SEED.svg?style=social&label=Star)](https://github.com/AILab-CVC/SEED)

+ [SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs](https://arxiv.org/pdf/2306.17842) (Jun 29, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.17842)

+ [Designing a Better Asymmetric VQGAN for StableDiffusion](https://arxiv.org/pdf/2306.04632) (Jun 7, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.04632)
  [![Star](https://img.shields.io/github/stars/buxiangzhiren/Asymmetric_VQGAN.svg?style=social&label=Star)](https://github.com/buxiangzhiren/Asymmetric_VQGAN)

+ [Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation](https://arxiv.org/pdf/2305.13607) (May 23, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.13607)
  [![Star](https://img.shields.io/github/stars/CrossmodalGroup/MaskedVectorQuantization.svg?style=social&label=Star)](https://github.com/CrossmodalGroup/MaskedVectorQuantization)

+ [Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization](https://arxiv.org/pdf/2305.11718) (May 19, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.11718)
  [![Star](https://img.shields.io/github/stars/CrossmodalGroup/DynamicVectorQuantization.svg?style=social&label=Star)](https://github.com/CrossmodalGroup/DynamicVectorQuantization)

+ [Binary Latent Diffusion](https://arxiv.org/pdf/2304.04820) (Apr 10, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.04820)
  [![Star](https://img.shields.io/github/stars/ZeWang95/BinaryLatentDiffusion.svg?style=social&label=Star)](https://github.com/ZeWang95/BinaryLatentDiffusion)

+ [Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment](https://arxiv.org/pdf/2302.00902) (Feb 2, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.00902)
  [![Star](https://img.shields.io/github/stars/haoliuhl/language-quantized-autoencoders.svg?style=social&label=Star)](https://github.com/haoliuhl/language-quantized-autoencoders)

+ [Muse: Text-To-Image Generation via Masked Generative Transformers](https://arxiv.org/pdf/2301.00704) (Jan 2, 2023. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.00704)
  [![Star](https://img.shields.io/github/stars/huggingface/open-muse.svg?style=social&label=Star)](https://github.com/huggingface/open-muse)

+ [Rethinking the Objectives of Vector-Quantized Tokenizers for Image Synthesis](https://arxiv.org/pdf/2212.03185) (Dec 6, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.03185)
  [![Star](https://img.shields.io/github/stars/TencentARC/BasicVQ-GEN.svg?style=social&label=Star)](https://github.com/TencentARC/BasicVQ-GEN)

+ [MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis](https://arxiv.org/pdf/2211.09117) (Nov 16, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.09117)
  [![Star](https://img.shields.io/github/stars/LTH14/mage.svg?style=social&label=Star)](https://github.com/LTH14/mage)

+ [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/pdf/2210.02399) (Oct 5, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02399)
  [![Star](https://img.shields.io/github/stars/lucidrains/phenaki-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/phenaki-pytorch)

+ [MoVQ: Modulating Quantized Vectors for High-Fidelity Image Generation](https://arxiv.org/pdf/2209.09002) (Sep 19, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.09002)
  [![Star](https://img.shields.io/github/stars/ai-forever/MoVQGAN.svg?style=social&label=Star)](https://github.com/ai-forever/MoVQGAN)

+ [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/pdf/2208.06366) (Aug 12, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.06366)
  [![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)](https://github.com/microsoft/unilm)

+ [DiVAE: Photorealistic Images Synthesis with Denoising Diffusion Decoder](https://arxiv.org/pdf/2206.00386) (Jun 1, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.00386)

+ [SQ-VAE: Variational Bayes on Discrete Representation with Self-annealed Stochastic Quantization](https://arxiv.org/pdf/2205.07547) (May 16, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.07547)
  [![Star](https://img.shields.io/github/stars/sony/sqvae.svg?style=social&label=Star)](https://github.com/sony/sqvae)

+ [CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers](https://arxiv.org/pdf/2204.14217) (Apr 28, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.14217)
  [![Star](https://img.shields.io/github/stars/THUDM/CogView2.svg?style=social&label=Star)](https://github.com/THUDM/CogView2)

+ [Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer](https://arxiv.org/pdf/2204.03638) (Apr 7, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.03638)
  [![Star](https://img.shields.io/github/stars/SongweiGe/TATS.svg?style=social&label=Star)](https://github.com/SongweiGe/TATS)

+ [Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors](https://arxiv.org/pdf/2203.13131) (Mar 24, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.13131)
  [![Star](https://img.shields.io/github/stars/CasualGANPapers/Make-A-Scene.svg?style=social&label=Star)](https://github.com/CasualGANPapers/Make-A-Scene)

+ [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/pdf/2203.01941) (Mar 3, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.01941)
  [![Star](https://img.shields.io/github/stars/kakaobrain/rq-vae-transformer.svg?style=social&label=Star)](https://github.com/kakaobrain/rq-vae-transformer)

+ [NÜWA-LIP: Language Guided Image Inpainting with Defect-free VQGAN](https://arxiv.org/pdf/2202.05009) (Feb 10, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.05009)
  [![Star](https://img.shields.io/github/stars/kodenii/NUWA-LIP.svg?style=social&label=Star)](https://github.com/kodenii/NUWA-LIP)

+ [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/pdf/2202.04200) (Feb 8, 2022. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.04200)
  [![Star](https://img.shields.io/github/stars/google-research/maskgit.svg?style=social&label=Star)](https://github.com/google-research/maskgit)
  [![Star](https://img.shields.io/github/stars/dome272/MaskGIT-pytorch.svg?style=social&label=Star)](https://github.com/dome272/MaskGIT-pytorch)

+ [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/pdf/2111.14822) (Nov 29, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.14822)
  [![Star](https://img.shields.io/github/stars/microsoft/VQ-Diffusion.svg?style=social&label=Star)](https://github.com/microsoft/VQ-Diffusion)

+ [Vector-quantized Image Modeling with Improved VQGAN](https://arxiv.org/pdf/2110.04627) (Oct 9, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2110.04627)
  [![Star](https://img.shields.io/github/stars/thuanz123/enhancing-transformers.svg?style=social&label=Star)](https://github.com/thuanz123/enhancing-transformers)

+ [CogView: Mastering Text-to-Image Generation via Transformers](https://arxiv.org/pdf/2105.13290) (May 26, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.13290)
  [![Star](https://img.shields.io/github/stars/THUDM/CogView.svg?style=social&label=Star)](https://github.com/THUDM/CogView)

+ [VideoGPT: Video Generation using VQ-VAE and Transformers](https://arxiv.org/pdf/2104.10157) (Apr 20, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.10157)
  [![Star](https://img.shields.io/github/stars/wilson1yan/VideoGPT.svg?style=social&label=Star)](https://github.com/wilson1yan/VideoGPT)

+ [Predicting Video with VQVAE](https://arxiv.org/pdf/2103.01950) (Mar 2, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2103.01950)
  [![Star](https://img.shields.io/github/stars/mattiasxu/Video-VQVAE.svg?style=social&label=Star)](https://github.com/mattiasxu/Video-VQVAE)

+ [Zero-Shot Text-to-Image Generation](https://arxiv.org/pdf/2102.12092) (Feb 24, 2021. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.12092)
  [![Star](https://img.shields.io/github/stars/openai/DALL-E.svg?style=social&label=Star)](https://github.com/openai/DALL-E)

+ [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2012.09841) (Dec 17, 2020. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2012.09841)
  [![Star](https://img.shields.io/github/stars/CompVis/taming-transformers.svg?style=social&label=Star)](https://github.com/CompVis/taming-transformers)

+ [Hierarchical Quantized Autoencoders](https://arxiv.org/pdf/2002.08111) (Feb 19, 2020. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2002.08111)
  [![Star](https://img.shields.io/github/stars/speechmatics/hqa.svg?style=social&label=Star)](https://github.com/speechmatics/hqa)

+ [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/pdf/1906.00446) (Jun 2, 2019. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1906.00446)
  [![Star](https://img.shields.io/github/stars/rosinality/vq-vae-2-pytorch.svg?style=social&label=Star)](https://github.com/rosinality/vq-vae-2-pytorch)

+ [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937) (Nov 2, 2017. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1711.00937)
  [![Star](https://img.shields.io/github/stars/MishaLaskin/vqvae.svg?style=social&label=Star)](https://github.com/MishaLaskin/vqvae)
  [![Star](https://img.shields.io/github/stars/lucidrains/vector-quantize-pytorch.svg?style=social&label=Star)](https://github.com/lucidrains/vector-quantize-pytorch)

### Hybrid

+ [OneVAE: Joint Discrete and Continuous Optimization Helps Discrete Video VAE Train Better](https://arxiv.org/pdf/2508.09857) (Aug 13, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.09857)
  [![Star](https://img.shields.io/github/stars/HVision-NKU/OneVAE.svg?style=social&label=Star)](https://github.com/HVision-NKU/OneVAE)

+ [DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](https://arxiv.org/pdf/2507.04947) (Jul 7, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.04947)
  [![Star](https://img.shields.io/github/stars/dc-ai-projects/DC-AR.svg?style=social&label=Star)](https://github.com/dc-ai-projects/DC-AR)

+ [TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation](https://arxiv.org/pdf/2505.05422) (May 8, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.05422)
  [![Star](https://img.shields.io/github/stars/TencentARC/TokLIP.svg?style=social&label=Star)](https://github.com/TencentARC/TokLIP)

+ [UniToken: Harmonizing Multimodal Understanding and Generation through Unified Visual Encoding](https://arxiv.org/pdf/2504.04423) (Apr 6, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.04423)
  [![Star](https://img.shields.io/github/stars/SxJyJay/UniToken.svg?style=social&label=Star)](https://github.com/SxJyJay/UniToken)

+ [Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens](https://arxiv.org/pdf/2501.07730) (Jan 13, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.07730)
  [![Star](https://img.shields.io/github/stars/bytedance/1d-tokenizer.svg?style=social&label=Star)](https://github.com/bytedance/1d-tokenizer)

+ [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/pdf/2501.03575) (Jan 7, 2025. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.03575)
  [![Star](https://img.shields.io/github/stars/NVIDIA/Cosmos-Tokenizer.svg?style=social&label=Star)](https://github.com/NVIDIA/Cosmos-Tokenizer)

+ [VidTok: A Versatile and Open-Source Video Tokenizer](https://arxiv.org/pdf/2412.13061) (Dec 17, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.13061)
  [![Star](https://img.shields.io/github/stars/microsoft/vidtok.svg?style=social&label=Star)](https://github.com/microsoft/vidtok)

+ [Language-Guided Image Tokenization for Generation](https://arxiv.org/pdf/2412.05796) (Dec 8, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.05796)

+ [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://arxiv.org/pdf/2410.10812) (Oct 14, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.10812)
  [![Star](https://img.shields.io/github/stars/mit-han-lab/hart.svg?style=social&label=Star)](https://github.com/mit-han-lab/hart)

+ [OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation](https://arxiv.org/pdf/2406.09399) (Jun 13, 2024. arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.09399)
  [![Star](https://img.shields.io/github/stars/FoundationVision/OmniTokenizer.svg?style=social&label=Star)](https://github.com/FoundationVision/OmniTokenizer)

## Acknowledgements

This template is provided by [Awesome-Unified-Multimodal-Models](https://github.com/showlab/Awesome-Unified-Multimodal-Models).
