# JPEG Robustness of AI-generated Image Detectors

Machine Learning and Programming — Yonsei University, Spring 2026  
Rachel Lu · Bi Yueqi

## Overview

This project investigates how JPEG compression affects the performance of AI-generated image (fake image) detectors, and whether XAI-guided analysis can diagnose the failure mechanism and inform robustness interventions.

## Research Questions

| # | Question |
|---|----------|
| RQ1 | Does JPEG compression cause fake recall drop across architectures? |
| RQ2 | How do fake-to-real prediction shifts occur at the per-image level? |
| RQ3 | Does XAI support a fragile cue reliance hypothesis? |
| RQ4 | Can background suppression or LazyStrike-inspired intervention improve JPEG robustness? |

## Method

### Stage 1: ResNet18 XAI-guided Pipeline
- Train ResNet18 baseline and observe JPEG-induced fake recall drop
- Use Grad-CAM to diagnose failure cases (heatmap correlation = 0.97)
- Apply Background Suppression using DeepLabV3 pseudo masks
- Result: JPEG fake recall improved from 81.33% to 86.67%

### Stage 2: Cross-architecture Comparison & Intervention Transfer
- Expand to ViT-B/16 and DINOv3-ViT-B/16 on the full dataset (~20K images)
- Paired prediction analysis to quantify fake-to-real shifts
- Test whether BG Suppression and LazyStrike-k1 transfer to ViT

## Key Results

### Baseline Comparison (12,005 test samples)

| Model | Orig Recall | JPEG Recall | Recall Drop | TP→FN |
|-------|------------|-------------|-------------|-------|
| ResNet18 | 93.45% | 91.12% | 2.33% | 144 |
| ViT-B/16 | 93.78% | 92.40% | 1.38% | 102 |
| DINOv3-ViT-B/16 | 97.45% | 93.33% | 4.12% | 247 |

### Intervention Transfer (ViT-B/16)

| Intervention | Orig Recall | JPEG Recall | Drop | TP→FN |
|-------------|------------|-------------|------|-------|
| Baseline | 93.78% | 92.40% | 1.38% | 102 |
| + BG Suppression | 95.50% | 92.50% | 3.00% | 183 |
| + LazyStrike-k1 | 95.63% | 93.57% | 2.07% | 136 |

### Key Findings
1. JPEG compression causes fake recall drop across all architectures
2. DINOv3 achieves the strongest clean performance but is the most fragile under compression
3. Background suppression works on ResNet18 but does not directly transfer to ViT
4. Robustness intervention must be architecture-aware

## XAI Cases

Selected TP→FN visualization panels are in `results/xai_cases/`. Each panel shows original image, JPEG image, Grad-CAM overlays, and prediction probabilities.

| Model | Case ID | Description |
|-------|---------|-------------|
| ResNet18 | 0882 | Bear — high-frequency fur texture destroyed |
| ResNet18 | 2354 | Desert house — structural edge degradation |
| ViT-B/16 | 2839 | Lake scene — patch-level texture dependency |
| ViT-B/16 | 0279 | Crowd — complex background reliance |
| DINOv3 | 0708 | Sports car — background bokeh/lighting (prob_fake: 0.998→0.005) |
| DINOv3 | 4851 | Male face — skin/hair detail compression |
| DINOv3 | 1806 | Face in flames — high-frequency + lighting artifacts |

## Repository Structure

```
├── README.md
├── notebooks/
│   ├── resnet18_baseline.ipynb
│   ├── resnet18_bg_suppression.ipynb
│   ├── vit_b16_baseline.ipynb
│   ├── dinov3_vit_b16_baseline.ipynb
│   ├── vit_lazystrike_k1.ipynb
│   ├── vit_bg_suppression.ipynb
│   ├── paired_analysis.ipynb
│   ├── bg_suppression_paired_analysis.ipynb
│   └── lazystrike_paired_analysis.ipynb
├── results/
│   ├── key_tables/
│   └── xai_cases/
└── slides/
```

## Notes

- Model checkpoints and full datasets are not included due to file size
- Notebooks were originally developed in Google Colab with GPU runtime
- JPEG compression applied at quality factor 30 (q30) using PIL/Pillow
