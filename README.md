Let me gather resources to verify feasibility before writing the proposal.













I have enough research data now. Here is the complete proposal:

---

# Project Proposal: Error-Aware Contrastive-Conditioned Diffusion Transformers for Image Variation Generation

## 1-Sentence Overall Idea
Investigate whether combining error-aware training (adapted from Matrix Game 3.0's residual-injection self-correction) with contrastive-MAE-pretrained image encoders (replacing class-label conditioning) measurably improves a small Diffusion Transformer's image variation quality, measured by FID and CLIP score on an ImageNet subset.

## Background
Diffusion Transformers (DiT; Peebles & Xie, 2023) established that replacing the U-Net backbone in latent diffusion with a Vision Transformer yields state-of-the-art class-conditional image generation on ImageNet. Standard DiT conditions on class labels via an embedding lookup fed through adaptive layer norm (adaLN-Zero). Two recent lines of work suggest complementary improvements:

1. **Error-Aware Training (Matrix Game 3.0, Wang et al. 2025):** In autoregressive diffusion video generation, the model sees only clean ground-truth frames during training but must condition on its own imperfect predictions at inference, causing error accumulation. Matrix Game 3.0 addresses this by storing the model's prediction residuals (`R = x_hat - x`) in an error buffer and re-injecting them into training inputs: `x_tilde = x + alpha * R`. The model then must still predict correct outputs given corrupted inputs, learning self-correction. This is conceptually related to Scheduled Sampling (Bengio et al., 2015) but uses the model's own structured errors rather than random corruption -- making it more realistic. This technique has **not** been studied for image-conditioned DiTs.

2. **Contrastive Masked Autoencoders (CMAE/ViC-MAE):** Standard MAE pretraining (He et al., 2022) learns features via pixel reconstruction but provides no explicit incentive for the latent space to be semantically discriminative. CMAE (Huang et al., 2023) and ViC-MAE (Hernandez et al., 2023) add a supervised contrastive loss on the encoder's [CLS] token during MAE pretraining, producing features that are both reconstructive and discriminative. The ACMAE framework (Singh, CS231n 2025) further demonstrates that injecting coarse labels via contrastive alignment during MAE pretraining improves downstream tasks in low-data regimes.

**This project fuses both ideas** into an image-to-image DiT: (a) replace class-label conditioning with CMAE image-feature conditioning (given a reference image, generate a new variation), and (b) apply error-aware training so the model learns to generate high-quality outputs even when conditioning features are imperfect. The architecture is a standard DiT-S/2 where the class-embedding lookup is replaced by a frozen CMAE encoder + linear projection producing the adaLN conditioning vector.

## Falsifiable Hypothesis
> A DiT-S/2 (33M parameters) trained for 100K steps on ImageNette (10-class ImageNet subset, 256x256) with error-aware conditioning injection (alpha=0.3, FIFO buffer of size 512, injection probability p=0.5) will achieve FID-5K **at least 10% lower** (better) than the identical model trained without error-aware injection. Independently, replacing class-label conditioning with CMAE-pretrained ViT-B/16 image features will reduce FID-5K by **at least 5%**. The combination will yield a total FID reduction of **at most 12%** (partially redundant gains).

**Why this is non-trivial and falsifiable:**
- Error-aware injection could *degrade* FID if model residuals act as destructive noise rather than useful augmentation (the structured-error hypothesis could be wrong for images vs. video).
- CMAE features could *underperform* simple class embeddings if the 768-dim CLS token provides less class-separable conditioning than a learned 10-class embedding (information bottleneck).
- The predicted interaction effect (partial redundancy) could go either direction -- the gains could be additive or fully redundant.

## Metrics
- **FID-5K (primary):** Frechet Inception Distance over 5,000 generated samples vs. validation set. Lower is better. Measures joint quality + diversity.
- **Inception Score (secondary):** Over the same 5,000 samples. Higher is better. Measures per-sample quality and class separability.
- **CLIP Score (auxiliary):** Cosine similarity between CLIP embeddings of the reference image and the generated image. Measures semantic fidelity of the image-to-image mapping.
- **LPIPS (auxiliary):** Learned Perceptual Image Patch Similarity between reference and generated image. Ensures the model produces variations, not copies (moderate LPIPS is desired).
- **Statistical test:** Mean +/- std over 3 random seeds; paired t-test (p < 0.05) for primary FID comparisons.

## Methodology

### Innovation 1: Error-Aware Training for DiT (adapted from Matrix Game 3.0)
During training, the model's own denoising errors are collected and re-injected into the conditioning signal:

1. **Error Buffer:** Maintain a FIFO buffer `B` of size 512, storing latent-space prediction residuals.
2. **Residual Collection:** After each training step, run a single-step DDIM decode of the predicted noise to obtain predicted clean latent `z_hat`. Compute `R = z_hat - z_target`. Append `R` to buffer (buffer begins filling after a 1,000-step warmup).
3. **Error Injection:** With probability `p=0.5`, sample `R_i` from `B` and corrupt the conditioning vector: `c_tilde = c + alpha * proj(R_i)`, where `proj()` is a frozen random linear map from latent dim to conditioning dim (initialized once, not trained).
4. **Training:** The model must still predict the correct noise epsilon given corrupted conditioning, learning self-correction against its own failure modes.
5. **Ablation:** Sweep `alpha` in {0.1, 0.3, 0.5}.

### Innovation 2: CMAE-Pretrained Image Conditioning
Replace the class-label embedding with image-derived features from a contrastive MAE encoder:

1. **Initialize** from `facebook/vit-mae-base` (ViT-B/16, 86M params, MAE-pretrained on ImageNet-1K).
2. **Add contrastive head:** MLP projector (768 -> 256) on the [CLS] token. Apply SupConLoss using the 10 ImageNette class labels as coarse supervision.
3. **Fine-tune** the full encoder + contrastive head for 20 epochs on ImageNette (combined loss: `L = L_reconstruct + 0.1 * L_contrastive`).
4. **Freeze** the CMAE encoder after pretraining.
5. **Conditioning:** Extract [CLS] token (768-dim) from a reference image. Linear projection (768 -> 384) maps it to DiT-S/2's conditioning dimension, replacing the class embedding lookup in adaLN-Zero.
6. At training time, the reference image is a randomly sampled same-class image (not the target), ensuring the model generalizes rather than memorizes.

## Experimental Design

| Step | Task | Time (T4) | Time (H100) |
|------|------|-----------|-------------|
| 1 | **Data preparation:** Download ImageNette 320px via HuggingFace. Resize to 256x256. Pre-extract VAE latents using `stabilityai/sd-vae-ft-mse` for all train/val images (fast-DiT approach eliminates VAE forward pass during training). | 0.5 h | 0.1 h |
| 2 | **CMAE encoder fine-tuning:** Init from `facebook/vit-mae-base`. Add contrastive head. Train 20 epochs on ImageNette, batch 64, lr=1e-4, cosine schedule, AdamW. Justification: 20 epochs suffices because we start from a strong MAE-pretrained encoder and only need to align the contrastive head. | 2 h | 0.3 h |
| 3 | **Exp A -- Baseline DiT-S/2:** Train from scratch on ImageNette latents, class-conditional (10 classes), 100K steps, batch 128 (gradient accumulation), lr=1e-4, fp16 + gradient checkpointing. Justification: DiT-S/2 at 33M params requires ~7 GB VRAM with fast-DiT optimizations, fitting T4. | 7 h | 1 h |
| 4 | **Exp B -- Error-Aware DiT-S/2:** Same as Step 3 + error-aware training loop (alpha=0.3, buffer 512, p=0.5). Buffer fills after 1K warmup steps. Justification: structured residual injection from Matrix Game 3.0 teaches self-correction. | 8 h | 1.2 h |
| 5 | **Exp C -- CMAE-Conditioned DiT-S/2:** Same as Step 3 but replacing class embedding with frozen CMAE encoder features + linear projection. Reference = random same-class image. Justification: tests whether richer image features improve generation vs. discrete class labels. | 8 h | 1.2 h |
| 6 | **Exp D -- Combined (CMAE + Error-Aware):** Both innovations together. Same hyperparams as B and C. | 9 h | 1.3 h |
| 7 | **Evaluation:** Generate 5K images per model (DDPM 250 steps, cfg=1.5). Compute FID-5K, IS, CLIP Score, LPIPS. 3 seeds per config. | 4 h | 0.5 h |
| 8 | **Ablation on alpha:** Error-aware training with alpha in {0.1, 0.3, 0.5} using best conditioning from Steps 3-6, 50K steps each. | 10 h | 1.5 h |
| 9 | **Gaussian-noise control:** Same as Step 4 but inject random Gaussian noise of matched norm instead of model residuals. Tests whether improvement comes from *structured* errors vs. generic augmentation. | 8 h | 1.2 h |

## Baseline
1. **B1 -- Standard DiT-S/2 (primary control):** Class-conditional DiT-S/2 trained with standard training on ImageNette. No error-aware injection, no CMAE conditioning. Isolates both innovations.
2. **B2 -- Standard-MAE-Conditioned DiT-S/2:** Same as Exp C but using `facebook/vit-mae-base` without the contrastive fine-tuning step. Isolates the benefit of the contrastive loss.
3. **B3 -- Gaussian-Noise-Injected DiT-S/2:** Same as Exp B but replacing structured model residuals with random Gaussian noise of matched L2 norm. Isolates whether the *structure* of errors matters.

## Resources/Assets

### Models
| Model | ID / Source | Params | Disk |
|-------|-----------|--------|------|
| MAE ViT-B/16 (encoder init) | `facebook/vit-mae-base` | 86M | ~330 MB |
| Stable Diffusion VAE | `stabilityai/sd-vae-ft-mse` | 83M | ~335 MB |
| DiT-S/2 (trained from scratch) | via `chuanyangjin/fast-DiT` code | 33M | ~130 MB |

### Datasets
| Dataset | HuggingFace ID | Size | Details |
|---------|---------------|------|---------|
| ImageNette | `frgfm/imagenette` (or direct: `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz`) | ~1.5 GB | 10-class ImageNet subset, 320px, 9,469 train / 3,925 val |

### Code
| Repository | URL | What we use |
|-----------|-----|-------------|
| **fast-DiT** | https://github.com/chuanyangjin/fast-DiT | `train.py` (single-GPU training loop with fp16 + grad ckpt), `models.py` (DiT arch), `extract_features.py` (VAE pre-extraction) |
| **facebookresearch/DiT** | https://github.com/facebookresearch/DiT | `models.py` (reference DiT-S/2 arch), `sample.py` (DDPM sampling + CFG) |
| **facebookresearch/mae** | https://github.com/facebookresearch/mae | `models_mae.py` (MAE ViT definition), `main_pretrain.py` (pretraining loop for CMAE adaptation) |
| **ViC-MAE** | https://github.com/jeffhernandez1995/ViC-MAE | Reference for contrastive MAE objective implementation |
| **CMAE** | https://github.com/ZhichengHuang/CMAE | `models/cmae.py` (contrastive head + SupCon loss implementation) |

## Compute Estimation

| Phase | T4 (16 GB) | L4 (24 GB) | H100 (80 GB) |
|-------|-----------|-----------|-------------|
| VAE feature extraction | 0.5 h | 0.3 h | 0.1 h |
| CMAE encoder fine-tune | 2 h | 1.5 h | 0.3 h |
| DiT-S/2 x 1 run (100K steps) | 7 h | 5 h | 1 h |
| DiT-S/2 x 4 factorial configs | 32 h | 22 h | 4.7 h |
| FID evaluation (5K x 4 models x 3 seeds) | 4 h | 3 h | 0.5 h |
| alpha ablation (3 runs x 50K steps) | 10.5 h | 7.5 h | 1.5 h |
| Gaussian control run | 8 h | 5.5 h | 1.2 h |
| **Full experiment suite** | **~57 h (2 sessions)** | **~40 h (2 sessions)** | **~8.3 h (1 session)** |
| **Minimum viable (baseline + combined + eval)** | **~20 h (1 session)** | **~14 h (1 session)** | **~3 h** |

**VRAM budget (peak, fp16 + gradient checkpointing):**
- DiT-S/2 train only: ~4 GB (model+optimizer) + ~2 GB (activations) + ~1 GB (batch) = **~7 GB** -- fits T4
- + frozen CMAE encoder: +0.7 GB = **~7.7 GB** -- fits T4
- + VAE decode at eval: +0.7 GB = **~8.4 GB** -- fits T4

**Recommended strategy:**
- **T4:** Run minimum-viable experiment (Exp A baseline + Exp D combined + eval) in one 20h session. Run remaining factorial arms and ablations in a second session.
- **H100:** All experiments including ablations comfortably fit in a single ~8.3h session.

## Verification Note
- **`facebook/vit-mae-base`**: Public HuggingFace checkpoint, no auth required. Verified at https://huggingface.co/facebook/vit-mae-base
- **`stabilityai/sd-vae-ft-mse`**: Public HuggingFace checkpoint. Verified at https://huggingface.co/stabilityai/sd-vae-ft-mse
- **`frgfm/imagenette`**: Freely available HuggingFace dataset (no ImageNet access agreement needed). Verified at https://huggingface.co/datasets/frgfm/imagenette
- **fast-DiT repo**: Includes working `train.py` for single-GPU training with `accelerate launch --mixed_precision fp16`. Verified at https://github.com/chuanyangjin/fast-DiT
- **CMAE repo**: Includes model definitions and contrastive loss. Verified at https://github.com/ZhichengHuang/CMAE
- **ViC-MAE repo**: Includes pretraining + fine-tuning scripts. Verified at https://github.com/jeffhernandez1995/ViC-MAE
- All code repos include both model architecture files and training loop scripts.

### HF Models
- `facebook/vit-mae-base` -- https://huggingface.co/facebook/vit-mae-base
- `stabilityai/sd-vae-ft-mse` -- https://huggingface.co/stabilityai/sd-vae-ft-mse

### HF Datasets
- `frgfm/imagenette` -- https://huggingface.co/datasets/frgfm/imagenette

### GitHub Repos
- https://github.com/chuanyangjin/fast-DiT -- Optimized single-GPU DiT training (train loop + model + VAE extraction)
- https://github.com/facebookresearch/DiT -- Original DiT (model arch + sampling)
- https://github.com/facebookresearch/mae -- MAE (encoder architecture + pretraining)
- https://github.com/jeffhernandez1995/ViC-MAE -- ViC-MAE (contrastive MAE reference)
- https://github.com/ZhichengHuang/CMAE -- CMAE (contrastive head + SupCon loss)

### Papers
1. W. Peebles and S. Xie, "Scalable Diffusion Models with Transformers," *ICCV 2023*. https://arxiv.org/abs/2212.09748
2. Z. Wang et al., "Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory," 2025. https://arxiv.org/abs/2604.08995
3. K. He et al., "Masked Autoencoders Are Scalable Vision Learners," *CVPR 2022*. https://arxiv.org/abs/2111.06377
4. J. Hernandez et al., "ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders," 2023. https://arxiv.org/abs/2303.12001
5. Z. Huang et al., "Contrastive Masked Autoencoders are Stronger Vision Learners," *IEEE TPAMI 2023*. https://arxiv.org/abs/2207.13532
6. S. Bengio et al., "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks," *NeurIPS 2015*. https://arxiv.org/abs/1506.03099
7. R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR 2022*. https://arxiv.org/abs/2112.10752
8. K. Singh, "Adaptive Contrastive Masked Autoencoders for Structured Representation Learning," *CS

Here is the complete project proposal, condensed:

---

# Error-Aware Contrastive-Conditioned DiT for Image Variation Generation

## 1-Sentence Overall Idea
Test whether adapting error-aware training from Matrix Game 3.0 (residual self-correction) and contrastive-MAE image conditioning (replacing class labels) jointly improve a small Diffusion Transformer's image generation quality on an ImageNet subset.

## Background
**DiT** (Peebles & Xie, 2023) replaces U-Net with a transformer for latent diffusion, conditioning on class labels via adaLN-Zero. Two orthogonal improvements are unexplored for DiTs:

**Error-Aware Training** (Matrix Game 3.0): During autoregressive video generation, models accumulate errors because training uses clean inputs but inference uses the model's own imperfect outputs. MG3.0 stores prediction residuals `R = x_hat - x` in a buffer, then injects them back: `x_tilde = x + alpha*R`, forcing self-correction. Related to Scheduled Sampling (Bengio et al., 2015) but uses *structured* model errors, not random noise. Never applied to image-conditioned DiTs.

**Contrastive MAE** (CMAE/ViC-MAE): Standard MAE features lack discriminability. Adding supervised contrastive loss on the [CLS] token during pretraining yields features that are both reconstructive and semantically discriminative -- ideal for conditioning a generative model.

**Our fusion:** Replace class-label conditioning with frozen CMAE encoder features (making DiT image-to-image: reference image in, generated variation out), and apply error-aware training so the model self-corrects against imperfect conditioning.

## Falsifiable Hypothesis
> DiT-S/2 (33M params) trained 100K steps on ImageNette with error-aware injection (alpha=0.3, buffer=512) will achieve FID-5K **>=10% lower** than standard training. CMAE conditioning will independently reduce FID-5K by **>=5%**. The combination yields **<=12%** total reduction (partial redundancy).

**Non-trivial because:** structured residuals could act as destructive noise; CMAE 768-dim features could underperform a simple 10-class embedding; interaction could be fully redundant or additive.

## Metrics
- **FID-5K (primary):** Frechet Inception Distance, 5K samples vs. val set
- **Inception Score (secondary):** quality + class separability
- **CLIP Score (auxiliary):** semantic similarity between reference and generated image
- **LPIPS (auxiliary):** perceptual distance (moderate = desired diversity)
- **Statistics:** mean +/- std over 3 seeds, paired t-test p<0.05

## Methodology

**Innovation 1 -- Error-Aware Training:** Maintain FIFO buffer (size 512) of latent residuals. After each step, compute `R = z_hat - z_target`, store in buffer (after 1K warmup). With probability p=0.5, corrupt conditioning: `c_tilde = c + alpha*proj(R_i)`. Model must still predict correct noise. Sweep alpha in {0.1, 0.3, 0.5}.

**Innovation 2 -- CMAE Image Conditioning:** Init from `facebook/vit-mae-base`. Add contrastive head (MLP 768->256 + SupConLoss on 10 class labels). Fine-tune 20 epochs. Freeze encoder. Extract [CLS] token, project 768->384 to replace DiT-S/2's class embedding in adaLN-Zero. Reference image = random same-class image (not target).

## Experimental Design (2x2 factorial + controls)

| # | Config | Description |
|---|--------|-------------|
| A | Baseline | DiT-S/2, class conditioning, standard training |
| B | +Error-Aware | DiT-S/2, class conditioning, error-aware (alpha=0.3) |
| C | +CMAE | DiT-S/2, CMAE image conditioning, standard training |
| D | Combined | DiT-S/2, CMAE conditioning + error-aware |
| B2 | MAE control | Like C but standard MAE encoder (no contrastive head) |
| B3 | Gaussian control | Like B but random Gaussian noise (matched norm) instead of model residuals |

**Steps:** (1) Download ImageNette, resize 256x256, pre-extract VAE latents. (2) CMAE fine-tune 20 epochs. (3-6) Train configs A-D, 100K steps each, batch 128, lr=1e-4, fp16+grad ckpt. (7) Generate 5K images per model (DDPM 250 steps, cfg=1.5), compute all metrics x3 seeds. (8) Alpha ablation.

## Baseline
- **B1:** Standard class-conditional DiT-S/2 (primary control)
- **B2:** Standard-MAE-conditioned DiT-S/2 (isolates contrastive benefit)
- **B3:** Gaussian-noise-injected DiT-S/2 (isolates structured-error benefit)

## Resources/Assets

### Models
| Model | HF ID | Params | Size |
|-------|-------|--------|------|
| MAE ViT-B/16 | `facebook/vit-mae-base` | 86M | 330 MB |
| SD VAE | `stabilityai/sd-vae-ft-mse` | 83M | 335 MB |
| DiT-S/2 | Trained from scratch via fast-DiT | 33M | 130 MB |

### Datasets
| Dataset | HF ID | Size |
|---------|-------|------|
| ImageNette | `frgfm/imagenette` | 1.5 GB, 10 classes, 9.5K train / 3.9K val |

### Code
| Repo | URL | Usage |
|------|-----|-------|
| fast-DiT | https://github.com/chuanyangjin/fast-DiT | Single-GPU training loop, model def, VAE extraction |
| DiT | https://github.com/facebookresearch/DiT | Reference arch + sampling |
| MAE | https://github.com/facebookresearch/mae | Encoder arch + pretrain loop |
| ViC-MAE | https://github.com/jeffhernandez1995/ViC-MAE | Contrastive MAE reference |
| CMAE | https://github.com/ZhichengHuang/CMAE | Contrastive head + SupCon loss |

## Compute Estimation

| Phase | T4 (16GB) | L4 (24GB) | H100 (80GB) |
|-------|-----------|-----------|-------------|
| VAE extraction + CMAE fine-tune | 2.5 h | 1.8 h | 0.4 h |
| DiT-S/2 x4 configs (100K steps each) | 32 h | 22 h | 4.7 h |
| Evaluation (5K x 4 models x 3 seeds) | 4 h | 3 h | 0.5 h |
| Ablations + controls | 18 h | 13 h | 2.7 h |
| **Full suite** | **~57 h (2 sessions)** | **~40 h** | **~8.3 h** |
| **Minimum viable (A+D+eval)** | **~20 h (1 session)** | **~14 h** | **~3 h** |

**VRAM peak (fp16 + grad ckpt):** DiT-S/2 train ~7 GB; +frozen CMAE ~7.7 GB; +VAE eval ~8.4 GB. **All fit T4.**

## Verification Note
All resources are publicly accessible without special auth. `facebook/vit-mae-base`, `stabilityai/sd-vae-ft-mse`, and `frgfm/imagenette` are directly downloadable from HuggingFace. All GitHub repos include model architecture files AND training loop scripts.

### HF Models
- https://huggingface.co/facebook/vit-mae-base
- https://huggingface.co/stabilityai/sd-vae-ft-mse

### HF Datasets
- https://huggingface.co/datasets/frgfm/imagenette

### GitHub Repos
- https://github.com/chuanyangjin/fast-DiT
- https://github.com/facebookresearch/DiT
- https://github.com/facebookresearch/mae
- https://github.com/jeffhernandez1995/ViC-MAE
- https://github.com/ZhichengHuang/CMAE

### Papers
1. Peebles & Xie, "Scalable Diffusion Models with Transformers," ICCV 2023. https://arxiv.org/abs/2212.09748
2. Wang et al., "Matrix-Game 3.0," 2025. https://arxiv.org/abs/2604.08995
3. He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022. https://arxiv.org/abs/2111.06377
4. Hernandez et al., "ViC-MAE," 2023. https://arxiv.org/abs/2303.12001
5. Huang et al., "CMAE," IEEE TPAMI 2023. https://arxiv.org/abs/2207.13532
6. Bengio et al., "Scheduled Sampling," NeurIPS 2015. https://arxiv.org/abs/1506.03099
7. Rombach et al., "Latent Diffusion Models," CVPR 2022. https://arxiv.org/abs/2112.10752
8. Singh, "ACMAE," CS231n 2025. https://cs231n.stanford.edu/2025/papers/CS231N%20-%20ACMAE%20-%20Karan%20Singh.pdf



























Refined Final Project Proposal

2D games for Action-conditioned video generation, or generative world models, or Video prediction with diffusion


1-Sentence Overall Idea: 
Background: 
Falsifiable Hypothesis: 
Methodology: 
Experimental Design:
Baseline: 
Resources/Assets:
Models: 
Datasets: 
Code: 
Compute Estimation: 
Verification Note:
HF Models: 
HF Datasets: 
GitHub Repos: 
Papers: 

Error Aware Training
Source of Inspiration: Matrix Game 3.0 Paper
Relates to Autoregressive Generation of Transformers (Dit, Vit) 
Main Idea: While training the module will see imperfect inputs at inference time, training it on imperfect inputs too using its past residual 
General Steps
Step 1: Collect Errors (The “Error Buffer”)
Residual (R) = x_hat - x 
Step 2: Inject Errors back into training
When training the model on the next batch, instead of feeding it perfectly with clean past frames, take a real past frame and corrupt it with one of those stored errors
x_tilda = x + ALPHA * R    ALPHA = controls strength of corruption
Step 3: Force the model to predict correct future frames anyway
The model is still asked to predict the correct future frames, even though its inputs are now slightly corrupted 
self-correction: the model learns to ignore or smooth out small eros in its inputs rather than amplifying them 
The idea relates to Scheduled Sampling and other papers follow similar idea but Matrix Game 3.0 was the first to do it for diffusion based video generation 
The errors are not random, they are the actual kind of errors the model itself tends to make and this is more realistic than just adding Gaussian noise because it teaches the model to handle its own specific failure does 

Adaptive Contrastive Masked Autoencoders for Structured Representation Learning
Source of Inspiration: Adaptive Contrastive Masked Autoencoders for Structured Representation Learning
Main Idea: Use Contrastive Loss for encoder part and include extra term in objective 
Investigate whether injecting coarse metadata during masked autoencoder pretraining improves downstream fine-grained medical image classification in low-data regimes.
Evaluating the Impact of Coarse Label Injection on Masked Autoencoders for Specific Image Data
1-Sentence Overall Idea: Investigate whether injecting coarse metadata during masked autoencoder pretraining improves downstream fine-grained <specific datatset> image classification in low-data regimes.
Background: The ViC-MAE (Visual Contrastive Masked Autoencoder) framework demonstrates that injecting coarse labels during masked autoencoder (MAE) pretraining via a supervised contrastive loss significantly improves downstream performance in low-data regimes. This project applies that concept to medical image classification, where fine-grained labels are expensive but coarse metadatais often readily available.
Methodology: Pretrain a ViT using the ViC-MAE objective, applying a contrastive loss to the visible patches based on the coarse metadata. Fine-tune the pretrained model on a small subset of the dataset using fine-grained labels. Compare the performance against a standard MAE baseline trained only with reconstruction loss.
Baseline: A standard ViT-MAE trained solely with a pixel-level reconstruction loss (no contrastive head or coarse labels) and fine-tuned under identical conditions.
Models: facebook/vit-mae-base via Hugging Face.

Data-Efficient Masked Autoencoders
Investigate whether continued MAE pretraining on unlabeled <data specific> imagery outperforms standard ImageNet-pretrained transfer when fine-tuning on very limited labeled data.

Memory Consistency in Autoregressive Video Generation Under Extreme Masking
Evaluate the robustness of context buffers in autoregressive video generation models when faced with missing or dropped input frames.

Quantify the trade-off between VAE latent compression ratios and generated image quality (FID) when fine-tuning small Diffusion Transformers.



















The PDF was binary. Let me try the early readable chunks and also check the HKUST blog for more architecture details.

I have enough information from the abstract, project page, and ICLR metadata. Here's the updated table with Field-DiT added:

---

# Comprehensive Comparison of Game World Models (Updated)

## Architecture & Scale

| Model | Year | Backbone | Denoiser Type | Model Size | VAE / Tokenizer | Resolution |
|-------|------|----------|--------------|------------|----------------|------------|
| **DIAMOND** | 2024 | Conv U-Net | U-Net (EDM) | ~54M | None (raw pixels) | 84×84 grayscale |
| **Open-Oasis** | 2024 | DiT (16 blocks, d=1024) | Spatio-Temporal DiT | 500M | ViT-VAE (ViT-L/20) | 256×256 RGB |
| **Dreamer4-MC** | 2025 | MAE tokenizer + DiT | DiT | Undisclosed | MAE tokenizer | ~128×128 |
| **nicklashansen/dreamer4** | 2025 | Block-causal Transformer | Shortcut forcing | Undisclosed | Causal tokenizer | 128×128 |
| **PlayGen (PGG)** | 2024 | VAE + RNN-like DiT | DiT blocks + CNN readout | ~33M DiT + VAE | Custom VAE | 128×128 |
| **MarioVGG** | 2024 | CogVideoX (fine-tuned) | 3D DiT (text-to-video) | ~2B (CogVideoX-2B) | CogVideoX VAE | 480×320 |
| **COMBAT** | 2026 | DiT (16 blocks, d=2048) | Spatio-Temporal DiT | 1.2B DiT + 340M VAE | Multi-modal DCAE | 448×736 |
| **GameNGen** | 2024 | Stable Diffusion 1.4 | U-Net | ~860M | SD 1.4 VAE | 320×240 |
| **MaaG** | 2025 | PGG baseline + modules | DiT blocks + CNN readout | ~34M (33M DiT + 0.7M modules) | PGG VAE | 96–128×128 |
| **ActionParty** | 2026 | Wan2.1-1.3B (fine-tuned) | Video DiT + subject state tokens | 1.3B | Wan2.1 VAE (8× spatial) | 512×512 |
| **GameGen-X** | 2024 | MSDiT + InstructNet | Video DiT (foundation) + InstructNet adapter | Undisclosed (multi-B est.) | 3D Spatio-Temporal VAE | Up to 720p |
| **Field-DiT** | 2025 | Probabilistic Field DiT | Diffusion on continuous fields | 675M | None (field representation — no VAE) | Flexible (continuous) |
| **GAIA-1** | 2023 | Autoregressive Transformer | N/A (autoregressive) | 9B | Custom | 288×512 |

## Action Conditioning & Game Domain

| Model | Game Domain | Dim | Action Type | # Actions | Multi-Agent | Action Conditioning Method | Prediction Mode |
|-------|------------|-----|-------------|-----------|-------------|--------------------------|----------------|
| **DIAMOND** | Atari (26 games inc. MsPacman) | 2D | Discrete | 4–18/game | Single | Learned emb → Adaptive GroupNorm in U-Net | Next frame (AR) |
| **Open-Oasis** | Minecraft | 3D | Discrete (keyboard) | 25 | Single | One-hot → Linear → added to timestep emb → adaLN | Next frame (AR) |
| **Dreamer4-MC** | Minecraft | 3D | Discrete (keyboard) | ~25 | Single | Action emb in interleaved sequence | Next frame (AR) |
| **nicklashansen/dreamer4** | DMControl (30 tasks) | 3D | **Continuous** | Varies | Single | Action emb in interleaved sequence | Next frame (AR) |
| **PlayGen (PGG)** | Super Mario Bros, DOOM | 2D+3D | Discrete | 7 (Mario), ~8 (DOOM) | Single | Action emb → cross-attention in DiT | Next frame (AR + hidden state) |
| **MarioVGG** | Super Mario Bros | 2D | **Text prompt** | N/A | Single | Natural language via text encoder | Multi-frame clip |
| **COMBAT** | Tekken 3 | 3D | Discrete (multi-hot) | 8 buttons | Two-player | Multi-hot → dense emb + sinusoidal time → AdaLNZero | Next frame (AR) |
| **GameNGen** | DOOM | 3D | Discrete (keys) | ~8 | Single | Learned emb → cross-attention (replaces text cross-attn) | Next frame (AR) |
| **MaaG** | Traveler, Pong, Pac-Man | 2D | Discrete | 3–5/game | Single | Cross-attention in DiT (PGG baseline) | Next frame (AR + hidden state) |
| **ActionParty** | 46 Melting Pot games | 2D | Discrete | 25 (7 base + 18 interact) | **Multi (up to 7)** | Per-subject emb → masked cross-attn + RoPE spatial bias | Next frame (AR, T=5) |
| **GameGen-X** | Open-world AAA games (GTA-like, RPG) | 3D | **Multi-modal**: keyboard + text + video prompt | Keyboard bindings + text | Single | Keyboard → AdaNorm; Text → cross-attn; Video prompt → latent addition | Multi-frame clip (AR continuation) |
| **Field-DiT** | Super Mario Bros | 2D | Discrete (control actions) | Undisclosed | Single | Control actions as cross-modality condition to DiT | Next frame (AR via view-wise sampling) |
| **GAIA-1** | Autonomous driving | 3D | **Continuous** | 3 (steer/throttle/brake) | Single | Tokenized + interleaved | Next frame (AR) |

## Availability & Compute

| Model | Open Code | Open Weights | Training Compute | Real-Time? | Min GPU for Inference |
|-------|-----------|-------------|-----------------|------------|----------------------|
| **DIAMOND** | ✅ [GitHub](https://github.com/eloialonso/diamond) | ✅ [HuggingFace](https://huggingface.co/eloialonso/diamond) (26 games) | 1 consumer GPU | ✅ ~20 FPS | Consumer GPU |
| **Open-Oasis** | ✅ [GitHub](https://github.com/etched-ai/open-oasis) | ✅ [HuggingFace](https://huggingface.co/Etched/oasis-500m) | Multi-GPU (undisclosed) | ✅ 20 FPS | 1× 24GB GPU |
| **Dreamer4-MC** | ✅ [GitHub](https://github.com/IamCreateAI/Dreamerv4-MC) | ✅ [HuggingFace](https://huggingface.co/IamCreateAI/Dreamerv4-MC) | Undisclosed | ✅ Real-time | 1× GPU |
| **nicklashansen/dreamer4** | ✅ [GitHub](https://github.com/nicklashansen/dreamer4) | ✅ [HuggingFace](https://huggingface.co/nicklashansen/dreamer4) | 8× RTX 3090, ~72h | ✅ Interactive | 1× GPU (≥2GB) |
| **PlayGen (PGG)** | ✅ [GitHub](https://github.com/GreatX3/Playable-Game-Generation) | ✅ Google Drive | ~8× A100 (est.) | ✅ 20 FPS on RTX 2060 | RTX 2060 |
| **MarioVGG** | ✅ [GitHub](https://github.com/Virtual-Protocol/mario-videogamegen) | ✅ [HuggingFace](https://huggingface.co/virtuals-protocol/mario-videogamegen) | 1× RTX 4090 | ❌ Clip gen only | RTX 4090 |
| **COMBAT** | ❌ Not released | ❌ Not released | 8× H200 | ✅ (after distill) | N/A |
| **GameNGen** | ❌ Not released | ❌ Not released | 128× TPU-v5e | ✅ 20+ FPS (1 TPU) | N/A |
| **MaaG** | ❌ Not released (builds on PGG) | ❌ Not released | 8× A100 40GB, ~3 days | ✅ (inherits PGG) | Consumer GPU (est.) |
| **ActionParty** | ⏳ "Coming soon" ([GitHub](https://github.com/action-party/action-party)) | ❌ Not yet released | Multi-GPU (batch 64, ~87.5k steps) | ❌ (20 diffusion steps) | ~1× GPU w/ 1.3B |
| **GameGen-X** | ⚠️ Dataset only ([GitHub](https://github.com/GameGen-X/GameGen-X)) | ❌ Not released | 8× H100 | ❌ Clip gen | N/A |
| **Field-DiT** | ⚠️ Partial ([demo zip](https://kfmei.com/Field-DiT/super-mario-bros-1-1.zip)) | ❌ Not released | Undisclosed | Undisclosed | Undisclosed |
| **GAIA-1** | ❌ Not released | ❌ Not released | Undisclosed (massive) | Undisclosed | N/A |

---

## Key Notes on Field-DiT

- **Venue:** ICLR 2025, from JHU (Kangfu Mei, Mo Zhou, Vishal M. Patel)
- **Novel paradigm:** Models data as **probabilistic fields** — continuous functions over metric spaces (e.g., (x,y,t) → RGB). This eliminates the need for a VAE/tokenizer entirely.
- **Unified architecture:** The same 675M DiT backbone handles video, 3D view synthesis, and game generation with **different weights** but the same architecture, using modality-specific cross-conditions (text for video, camera poses for 3D, control actions for games)
- **View-wise sampling + autoregressive generation** enables long-context coherence that prior probabilistic field models lacked
- **Game demo:** Super Mario Bros (2D), with a downloadable demo zip on the project page. No full training code, no pretrained weights publicly released.
- **Relatively compact** at 675M — more feasible for single-GPU finetuning than multi-B models




Github repository of Game Generation papers: JingyeChen/awesome-game-generation


Papers not based on Video games 

AVID: Adapting Video Diffusion Models to World Models
Paper: https://arxiv.org/abs/2410.12822

PAN: A World Model for General, Interactable, and Long-Horizon World Simulation
Paper: https://arxiv.org/abs/2511.09057



Field-DiT (Diffusion Transformer on Unified Video, 3D, and Game) Field-DiT is an architecture designed to unify different visual tasks under a single Transformer backbone. It treats 2D game generation as a specialized case of video generation where the model is conditioned on control actions. (https://openreview.net/forum?id=w6YS9A78fq)


Models names are in Red

Playable Game Generation - PlayGen
Paper: https://arxiv.org/abs/2412.00887
GitHub: GreatX3/Playable-Game-Generation

Video Game Generation: A Practical Study using Mario - MarioVGG
Project page: https://virtual-protocol.github.io/mario-videogamegen/
Model: https://huggingface.co/virtuals-protocol/mario-videogamegen
Blog post: https://virtuals.substack.com/p/video-game-generation-a-practical

Diffusion for World Modeling: Visual Details Matter in Atari – DIAMOND
Paper: https://github.com/eloialonso/diamond
GitHub: https://github.com/eloialonso/diamond
Project page: https://diamond-wm.github.io/

Matrix Game 3.0 - Matrix Game 3.0
Paper: https://arxiv.org/abs/2604.08995
GitHub: https://github.com/SkyworkAI/Matrix-Game
Project page: https://matrix-game-v3.github.io/
Huggingface: https://huggingface.co/Skywork/Matrix-Game-3.0

Training Agents Inside of Scalable World Models – Dreamer 4
Paper: https://arxiv.org/abs/2509.24527
Project page: https://danijar.com/project/dreamer4/

(no name paper) – Open-Oasis 500M
GitHub: https://github.com/etched-ai/open-oasis
Huggingface: https://huggingface.co/Etched/oasis-500m

Model as a Game: On Numerical and Spatial Consistency for Generative Games – MaaG
Paper: https://arxiv.org/abs/2503.21172
Project page: https://www.microsoft.com/en-us/research/articles/maag-a-new-framework-for-consistent-ai-generated-games/?lang=ja
Blog post: https://joshuaberkowitz.us/blog/news-1/maag-model-as-a-game-is-solving-consistency-challenges-in-ai-generated-games-395

Diffusion Models Are Real-Time Game Engines – GameNGen
Paper: https://arxiv.org/abs/2408.14837
Project Page: https://gamengen.github.io/

COMBAT: Conditional World Models for Behavioral Agent Training – COMBAT
Paper: https://arxiv.org/abs/2603.00825

ActionParty: Multi-Subject Action Binding in Generative Video Games – ActionParty
Paper: https://arxiv.org/abs/2604.02330
Project page: https://action-party.github.io/

GameGen-X: Interactive Open-world Game Video Generation – GameGen-X
Paper: https://arxiv.org/abs/2411.00769
GitHub: https://github.com/GameGen-X/GameGen-X
Project Page: https://gamegen-x.github.io/
