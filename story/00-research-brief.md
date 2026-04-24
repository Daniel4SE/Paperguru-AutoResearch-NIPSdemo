# Auto-Research Brief: Rotation-Aware Gradient Estimation for VQ-VAEs

## Research Topic

**Rotation-Aware Gradient Estimation for Vector-Quantized Autoencoders**

通过旋转-缩放线性变换重构 VQ-VAE 量化层的梯度传播路径，在保持前向输出不变的前提下，使梯度反向传播时保留编码器输出与码本向量的角度关系，从而缓解 codebook collapse、提升重建质量与码本利用率。

核心改动发生在量化层：用一个"前向视为常量、反向参与梯度流"的 rotation + rescaling 变换 $\tilde{q} = R(z_e) \cdot \|q\|/\|z_e\| \cdot z_e$ 替代传统 Straight-Through Estimator (STE) 的恒等梯度复制。

## Research Questions

1. **RQ1**：相比 STE，rotation-based 梯度估计能否在相同码本规模下获得更低的重建误差？
2. **RQ2**：该方法能否显著提升 codebook utilization（有效码本向量占比），抑制 collapse？
3. **RQ3**：在下游生成任务（latent autoregressive / diffusion）中，更好的离散表示是否转化为更优的生成指标（FID / IS）？
4. **RQ4**：与 FSQ、Gumbel-Softmax、rotation trick 等替代量化方案相比，计算开销与性能的 trade-off 如何？

## Datasets

| 用途 | 数据集 | 分辨率 | 规模 | 评测指标 |
|---|---|---|---|---|
| 主实验（图像重建） | ImageNet-1k | 128×128 / 256×256 | 1.28M | rFID, PSNR, SSIM, LPIPS |
| 消融/快速迭代 | CIFAR-10 | 32×32 | 50k | rFID, bits-per-dim |
| 人脸高保真 | FFHQ | 256×256 | 70k | rFID, LPIPS |
| 下游生成验证 | ImageNet-1k (class-conditional) | 256×256 | 1.28M | gFID, IS, Precision/Recall |
| （可选）视频/3D | UCF-101 | - | - | rFID per frame |

**码本指标**（贯穿所有数据集）：
- Codebook usage (%)：训练后被至少激活一次的码本向量比例
- Codebook perplexity：$\exp(-\sum_k p_k \log p_k)$
- Active codes per batch

## Baselines

| 类别 | 方法 | 参考文献 | 对应 prompt 文献 |
|---|---|---|---|
| 原始离散 VAE | VQ-VAE (STE) | van den Oord et al., 2017 | *Neural discrete representation learning* |
| 生成式离散 | VQ-GAN | Esser et al., 2021 | *Taming transformers* (implied) |
| 改进 STE | VQ-STE++ | Huh et al., 2023 | *Straightening out the STE* |
| 无码本量化 | FSQ | Mentzer et al., 2023 | *Finite scalar quantization* |
| 改进 VQGAN | ViT-VQGAN | Yu et al., 2022 | *Vector-quantized image modeling* |
| 随机梯度估计 | Gumbel-Softmax VAE | Jang et al., 2017 | *Categorical reparameterization* |
| 连续 VAE 参照 | KL-VAE | Kingma & Welling, 2014 | *Auto-encoding variational bayes* |
| 应用基准 | LDM 的 VQ autoencoder | Rombach et al., 2022 | *High-resolution image synthesis* |
| 统一框架 | UViM | Kolesnikov et al., 2022 | *UViM* |

## Proposed Method 实现骨架

```
Encoder → z_e
        ↓
     nearest-neighbor lookup → q (codebook vector)
        ↓
     R = Householder(z_e, q)            # 反射矩阵，O(d) 计算
     s = ||q|| / ||z_e||                # 缩放因子
     q̃ = sg[s · R] · z_e                # 前向等于 q；反向保留角度
        ↓
     Decoder(q̃) → x̂
Loss = ||x - x̂||² + β·||sg[q] - z_e||²  + codebook EMA update
```

**关键实现要点**
- Householder 反射：$R = I - 2vv^\top$，$v = (z_e - q)/\|z_e - q\|$，避免显式存储 $d \times d$ 矩阵
- `sg[·]`（stop-gradient）作用于 rotation 和 rescaling，使其前向参与、反向视为常量
- 码本更新使用 EMA（$\gamma = 0.99$），配合 dead-code reinit

## 实验矩阵

| 实验 | 目的 | 变量 |
|---|---|---|
| E1 重建主表 | 对比 baselines | {VQ-VAE, VQ-STE++, FSQ, Ours} × {128, 256} 分辨率 |
| E2 码本规模消融 | RQ1/RQ2 | codebook size ∈ {1024, 4096, 8192, 16384} |
| E3 β 消融 | 超参敏感性 | β ∈ {0.25, 0.5, 1.0, 2.0} |
| E4 rotation 变体 | 方法内拆解 | {no rotation, rotation only, rotation+rescale} |
| E5 下游生成 | RQ3 | 冻结 tokenizer → 训练 autoregressive transformer on ImageNet |
| E6 计算开销 | RQ4 | wall-clock, FLOPs, peak memory |

## 关键超参（起始配置）

- Codebook size: `8192`，dim `8`（follow VQ-GAN / FSQ 的低维趋势）
- Commitment β: `0.25`
- EMA decay: `0.99`
- Optimizer: AdamW, lr `1e-4`, cosine schedule
- Batch size: `256`（ImageNet 256×256，8×A100）
- Training: 250k steps（主实验），50k steps（CIFAR 快速消融）

## Deliverables

1. `models/vq_rotation.py` — 量化层实现
2. `train_vqae.py` — 训练脚本（复用 VQ-GAN codebase）
3. `eval/` — rFID / codebook metrics / gFID
4. `configs/` — 所有实验 YAML
5. `results/` — 日志 + checkpoints + 图表
6. `paper/` — LaTeX 手稿

## Reference Papers

1. van den Oord et al. *Neural Discrete Representation Learning.* NeurIPS 2017.
2. Huh et al. *Straightening out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks.* ICML 2023.
3. Bengio et al. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv 2013.
4. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
5. Mentzer et al. *Finite Scalar Quantization: VQ-VAE Made Simple.* ICLR 2024.
6. Cover & Thomas. *Elements of Information Theory.* Wiley, 2006.
7. Yu et al. *Vector-Quantized Image Modeling with Improved VQGAN.* ICLR 2022.
8. Kolesnikov et al. *UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes.* NeurIPS 2022.
9. Kingma & Welling. *Auto-Encoding Variational Bayes.* ICLR 2014.
10. Jang et al. *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017.
