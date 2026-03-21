# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenClaw-RL is a fully asynchronous reinforcement learning framework for training personalized AI agents from natural conversation feedback. It has two tracks: (1) Personal Agent Optimization via Binary RL, On-Policy Distillation (OPD), or a Combined method, and (2) Scalable Agentic RL for terminal, GUI, SWE, and tool-call environments.

## Repository Structure

The repo is a monorepo with distinct subsystems:

- **`slime/`** — Base RL training framework (Python). Built on Ray + Megatron-LM. Contains backends, rollout workers, task routing, and utilities. This is the shared training infrastructure — **do not modify unless absolutely necessary**.
- **`Megatron-LM/`** — NVIDIA Megatron-LM fork used as the distributed training backend.
- **`openclaw/`** — TypeScript/Node.js runtime. OpenAI-compatible API gateway with multi-channel messaging (Telegram, Discord, Slack, etc.). Separate from the RL training code.
- **`openclaw-rl/`** — Binary RL (GRPO) method implementation.
- **`openclaw-opd/`** — On-Policy Distillation method implementation.
- **`openclaw-combine/`** — Combined RL + OPD method (recommended).
- **`openclaw-tinker/`** — Cloud deployment via Tinker (LoRA-only, no local GPU needed).
- **`openclaw-test/`** — Evaluation framework (student/teacher chat with GSM8K).
- **`terminal-rl/`**, **`gui-rl/`**, **`swe-rl/`**, **`toolcall-rl/`** — Track 2 agent implementations for real-world environments.
- **`src/megatron-core/`** — Megatron core mirror.

Each method folder (`openclaw-rl/`, `openclaw-opd/`, `openclaw-combine/`, etc.) is self-contained with its own launch scripts, API server, rollout logic, loss functions, and README.

## Environment Setup

```bash
# Python side (CUDA 12.9, Python 3.12)
conda create --name openclaw-rl python=3.12
pip install torch==2.9.1+cu129 torchvision==0.24.1+cu129 torchaudio==2.9.1+cu129 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install -e slime/

# Node.js side (openclaw runtime)
cd openclaw && pnpm install && pnpm build
```

See `instructions/README.md` for full setup including apex, flash-attn, and DeepEP.

## Common Commands

### Training (Python/RL)

```bash
# Run combined method (recommended) — execute from slime/ directory
cd slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh

# Run with LoRA
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine_lora.sh

# Binary RL only
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh

# OPD only
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh

# Cloud deployment via Tinker (no GPU)
cd openclaw-tinker
python run.py --method combine --model-name Qwen/Qwen3-8B --batch-size 16
```

### Python Linting & Testing

```bash
# Linting (slime)
ruff check slime/
black --check --line-length 119 slime/

# Tests (slime)
cd slime
pytest tests/
pytest -m unit tests/          # unit tests only
pytest -m integration tests/   # integration tests only
```

### OpenClaw Runtime (TypeScript)

```bash
cd openclaw
pnpm build                # Full build
pnpm check                # Lint + format + type check (oxlint + oxfmt + tsgo)
pnpm test                 # Full test suite (vitest)
pnpm test:fast            # Unit tests only
pnpm test:coverage        # Tests with coverage
pnpm dev                  # Dev server
```

## Architecture

### Async 4-Component Loop

The core training loop decouples four independent async processes:
1. **Serving** — Policy model served via SGLang as OpenAI-compatible API (port 30000)
2. **Rollout Collection** — Intercepts live multi-turn conversations, captures log-probabilities
3. **Evaluation** — PRM/judge scores responses asynchronously (majority voting)
4. **Training** — Consumes scored samples via Ray cluster, runs policy gradient updates

None block each other — the model serves requests while training runs in background.

### Method Extension Points

The slime framework exposes pluggable extension points used by each method folder:
- `--custom-loss-function-path` — Custom loss functions (e.g., `combine_loss.py`)
- `--custom-rm-path` — Custom reward models
- `--custom-generate-function-path` — Custom generation logic
- `--rollout-function-path` — Custom rollout logic

### Key Environment Variables

Training GPU partitioning: `NUM_GPUS`, `ACTOR_GPUS`, `ROLLOUT_GPUS`, `PRM_GPUS`
Model: `HF_CKPT` (HuggingFace checkpoint path)
Method-specific: `OPENCLAW_COMBINE_W_RL`, `OPENCLAW_COMBINE_W_OPD`, `PRM_M` (judge votes)

### Code Style

- **Python**: `black` (line length 119), `isort` (black profile), `ruff` (E/F/B/UP rules)
- **TypeScript**: `oxlint` (type-aware), `oxfmt` for formatting
- **Node.js**: pnpm 10.23.0, Node 22.12.0+, ES modules

## Contribution Conventions

- New methods go in new top-level folders parallel to existing ones (e.g., `openclaw-opd/`)
- Extending existing methods: add new files (new `.sh` scripts, etc.) rather than modifying existing ones
- Follow shell script conventions: GPU partitioning variables, `CKPT_ARGS`/`ROLLOUT_ARGS`/`OPTIMIZER_ARGS` patterns, `ray job submit` launch pattern
- Do not modify `slime/`, `Megatron-LM/`, or `openclaw/` core code — use extension points instead
