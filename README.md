
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
