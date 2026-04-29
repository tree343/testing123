



















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
