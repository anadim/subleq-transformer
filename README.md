# SUBLEQ Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anadim/subleq-transformer/blob/main/subleq_transformer_demo.ipynb)

Two independent approaches to making a standard transformer execute a Turing-complete computer — one with **hand-coded weights** (no training), one **learned from data** (no hand-coding).

Both work. Both achieve 100% single-step accuracy and near-perfect multi-step generalization.

<p align="center">
  <img src="subleq_demo.gif" alt="Transformer executing multiply(7,9) = 63" width="700">
</p>

## What is SUBLEQ?

SUBLEQ (**SU**btract and **B**ranch if **L**ess than or **EQ**ual to zero) is a one-instruction ISA that is Turing complete. Each instruction `(a, b, c)` does:

```
mem[b] -= mem[a]
if mem[b] <= 0: goto c
else: goto next instruction (pc + 3)
```

One instruction, and you can compute anything — addition, multiplication, sorting, anything a normal computer can do.

---

## Round 1: Hand-Coded (`round1_constructed/`)

A standard transformer with **analytically set weights** — every one of the 2.1M parameters is computed by hand, not learned.

### Architecture

| Parameter | Value |
|---|---|
| Layers | 4 (Pre-LN, ReLU activation, no causal mask) |
| Attention heads | 8 (d_head = 4) |
| d_model | 32 |
| d_ff | 64 |
| Parameters | **2,143,712** total (~100 nonzero in transformer logic) |
| Memory | 416 cells, 16-bit signed integers [-32768, 32767] |
| Vocabulary | 65,538 tokens (value + offset encoding) |
| Sequence length | 417 (1 PC + 416 memory cells) |

### How It Works

The 32-dimensional residual stream acts as a register file with ~30 named dimensions. Four layers implement a complete SUBLEQ step:

| Layer | Function | Mechanism |
|-------|----------|-----------|
| 1 | Read instruction (a, b, c) from mem[pc..pc+2] | Content-based addressing via `q·k = -s(k-t)²` |
| 2 | Fetch mem[a] and mem[b], compute subtraction | Second pointer dereference + ReLU arithmetic |
| 3 | Broadcast write address, build position indicator | Broadcast attention + hat function `ReLU(j-b) - 2·ReLU(j-b-1) + ReLU(j-b-2)` |
| 4 | Write result to exactly one cell, update PC | Binary MUX: `s·z = ½[ReLU(z+2Ms-M) - ReLU(-z+2Ms-M)]` |

### Test Results (2,087 tests)

| Tier | Test | Count | Accuracy |
|------|------|-------|----------|
| 1 | Negate (v in [-100, 100]) | 201 | **100%** (201/201) |
| 2 | Addition (a,b in [-10, 10]) | 441 | **100%** (441/441) |
| 3 | Multiply (10 pairs) | 10 | **100%** (10/10) |
| 4 | Random single-step | 1,200 | **100%** (1,200/1,200) |
| 5 | Random multi-step (up to 200 steps) | 200 | **77.5%** (155/200) |
| 6 | Bubble sort (n=2..8) | 35 | **100%** (35/35) |
| | **Total** | **2,087** | **97.8%** (2,042/2,087) |

**100% on all structured programs.** The 45 errors in Tier 5 are from random multi-step programs where small per-step errors compound over 200 iterative steps — the model's single-step accuracy is perfect.

### Usage

```bash
cd round1_constructed
python demo.py    # Watch step-by-step execution
python eval.py    # Full 2,087-test suite
```

---

## Round 2: Trained (`round2_trained/`)

A standard transformer **trained from scratch** on single-step SUBLEQ state transitions — then it generalizes to run multi-step programs (Fibonacci, multiplication, division, square root) it never saw executed as full programs during training.

### Architecture

| Parameter | Value |
|---|---|
| Layers | 6 (Pre-LN, GELU activation, bidirectional) |
| Attention heads | 8 (d_head = 32) |
| d_model | 256 |
| d_ff | 1,024 |
| Parameters | **4,879,360** |
| Memory | 32 cells, 8-bit signed integers [-128, 127] |
| Vocabulary | 256 tokens (byte-level, two's complement) |
| Sequence length | 33 (1 PC + 32 memory cells) |
| Embeddings | Token + Position + Type (PC vs memory) |

### Training

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, warmup 1K steps, cosine decay) |
| Total steps | 80,000 |
| Batch size | 256 |
| Data | 100K examples, regenerated every 2K steps |
| Loss | Weighted cross-entropy (100x on changed positions, 1x on unchanged) |
| Curriculum | 4 phases: 1-2 instr (0-8K) → 1-4 (8-20K) → 1-6 (20-36K) → 1-8 (36-80K) |
| Hardware | Single Apple Silicon laptop (MPS backend), ~4 hours |

**Training data composition**: 60% random single-step pairs + 40% execution traces (negate, addition, countdown, multiply, random programs). No multi-step programs like Fibonacci, division, or square root appear in training — these are emergent.

<p align="center">
  <img src="fig1_training_curve.png" alt="Training curve with curriculum phases" width="700">
</p>

### Test Results

**Single-step accuracy** (3,682 held-out states after filtering halts):

| Instructions | Full accuracy | Changed-position accuracy |
|---|---|---|
| 1-8 (all) | 100.0% | 100.0% |

100% on every instruction count. Zero errors.

**Multi-step programs** (never seen as full program executions during training):

| Program | Cases | Max steps | Accuracy | Notes |
|---------|-------|-----------|----------|-------|
| Negate | 201 | 3 | 100% | |
| Addition | 300 | 4 | 100% | |
| Countdown | 20 | 39 | 100% | |
| Multiply | 141 | ~33 | **100%** | Single steps in training; full programs never seen |
| Fibonacci | 6 | 47 | **100%** | Never in training data |
| Division | 16 | 91 | **93.8%** | Never in training data; div(126,7) fails |
| Square root | 20 | 61 | **95.0%** | Never in training data; isqrt(120) fails |
| Random programs | 100 | 30 | **100%** | |
| **Total** | **804** | | **99.8%** | |

The longest correct computation: isqrt(100) = 10, requiring **61 consecutive correct steps** with zero errors.

<p align="center">
  <img src="fig3_multiplication_table.png" alt="12x12 multiplication table — 141/141 correct" width="600">
</p>

### Usage

```bash
cd round2_trained
python train.py              # Train from scratch (~4 hours on MPS)
python eval.py               # Single-step + multi-step evaluation
python demo.py               # Fibonacci, multiplication, division, square root
python play.py               # Interactive REPL
```

---

## Quickstart

**Try it now** — no install needed: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anadim/subleq-transformer/blob/main/subleq_transformer_demo.ipynb)

Or run locally:

```bash
git clone https://github.com/anadim/subleq-transformer.git
cd subleq-transformer
pip install torch

# Round 1: No training needed — just run
cd round1_constructed && python demo.py

# Round 2: Train from scratch (or use provided checkpoint)
cd ../round2_trained && python train.py && python demo.py
```

## Repository Structure

```
subleq-transformer/
├── README.md               # This file
├── LICENSE                  # MIT
├── requirements.txt        # torch>=2.0.0
│
├── round1_constructed/     # Hand-coded transformer (no training)
│   ├── model.py            # 2.1M-param transformer with analytical weights
│   ├── interpreter.py      # SUBLEQ interpreter (416 cells, 16-bit)
│   ├── programs.py         # Programs (negate, add, multiply, bubble sort)
│   ├── demo.py             # Step-by-step execution demos
│   ├── eval.py             # Full 2,087-test suite
│   └── report.pdf          # Technical report on the construction
│
├── round2_trained/         # Trained transformer
│   ├── subleq/             # Python package (interpreter, tokenizer, model, data)
│   ├── train.py            # Training script (curriculum, weighted loss)
│   ├── eval.py             # Evaluation (single-step + multi-step)
│   ├── demo.py             # ANSI-colored program demos
│   ├── play.py             # Interactive REPL
│   ├── Makefile            # make train, eval, demo, play
│   └── figures/            # Training curves, rollout visualizations
│
└── paper/                  # Academic paper (LaTeX + PDF)
    ├── paper.tex
    └── paper.pdf
```

## Key Insights

**Round 1** shows that a standard transformer *can* implement a computer — it's not a question of learning capacity but of representational capacity. The construction reveals:
- Attention can dereference pointers via quadratic identity: `q·k = -s(k-t)² + const`
- ReLU FFNs compute perfect integer step functions: `1[x>0] = ReLU(x) - ReLU(x-1)`
- Selective memory writes require 2 FFN layers (hat function → MUX), explaining why 4 layers are needed
- Only ~100 of 2.1M weights are nonzero — the rest are structural zeros

**Round 2** shows that a transformer *learns* to implement a computer from data — trained only on single-step transitions, it discovers how to chain steps into arbitrary-length programs. The emergent multi-step generalization includes:
- Fibonacci sequences (up to F(13) = 127, 47 steps) — never in training data
- Full 12×12 multiplication table (141/141 correct)
- Integer division (15/16 correct) and square root (19/20 correct) — never in training data
- 802/804 multi-step test instances correct overall (99.8%)

**Width > Depth**: The wide model (d=256, 6 layers, 4.9M params) achieves 100% while a deep model (d=128, 12 layers, 2.4M params) plateaus at 74.8%. Information routing bandwidth (d_head) is the bottleneck, not computational depth.

<p align="center">
  <img src="fig2_width_vs_depth.png" alt="Width dominates depth for SUBLEQ execution" width="650">
</p>

## Citation

```bibtex
@misc{subleq-transformer,
  title={A Transformer That Executes a One-Instruction Computer},
  year={2026}
}
```
