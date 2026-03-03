# Round 2: Trained SUBLEQ Transformer

A 4.9M-parameter transformer **trained from scratch** to execute SUBLEQ — then it generalizes to run programs (Fibonacci, multiplication, division, square root) it never saw executed as full programs during training.

## Architecture

| Parameter | Value |
|---|---|
| Type | Encoder-only (bidirectional), Pre-LayerNorm |
| Layers | 6 |
| Attention heads | 8 (d_head = 32) |
| d_model | 256 |
| d_ff | 1,024 |
| Activation | GELU |
| Parameters | **4,879,360** |
| Memory | 32 cells, 8-bit signed integers [-128, 127] |
| Vocabulary | 256 tokens (byte-level, two's complement) |
| Sequence length | 33 (1 PC + 32 memory cells) |
| Embeddings | Token (256×256) + Position (33×256) + Type (2×256, PC vs memory) |
| Initialization | N(0, 0.02), LayerNorm weight=1, bias=0 |

## Training Recipe

| Parameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, weight_decay=0.01) |
| Learning rate | 3×10⁻⁴ peak, linear warmup 1K steps, cosine decay to 0 |
| Gradient clipping | Max norm 1.0 |
| Total steps | 80,000 |
| Batch size | 256 |
| Training examples | 100,000 per phase, regenerated every 2,000 steps |
| Loss | Weighted cross-entropy: **100× on changed positions**, 1× on unchanged |
| Hardware | Single Apple Silicon laptop (MPS), ~4 hours |

### Curriculum Schedule

| Phase | Steps | Instruction range | Training budget |
|-------|-------|-------------------|-----------------|
| 1 | 0–8K | 1–2 instructions | 10% |
| 2 | 8K–20K | 1–4 instructions | 15% |
| 3 | 20K–36K | 1–6 instructions | 20% |
| 4 | 36K–80K | 1–8 instructions | 55% |

### Training Data Composition

Each phase generates 100K examples:
- **60% random single-step pairs**: random valid SUBLEQ state → execute one step → (input, output) pair
- **40% execution traces**: multi-step programs, each consecutive pair becomes a training example
  - 15% negate (v ∈ [-100, 100])
  - 15% addition (a, b ∈ [-60, 60])
  - 15% countdown (n ∈ [1, 20])
  - 10% multiplication (a ∈ [1, 10])
  - 45% random programs (1–k instructions, k = current curriculum max)

**Key**: No Fibonacci, division, or square root programs appear in training — these are **emergent capabilities**.

### Why 100× Loss Weighting?

In a 33-position sequence, only ~2 positions change per step (PC and the modified cell). Without weighting, a model that copies input unchanged gets ~94% per-position accuracy while learning nothing. The 100:1 weight forces the model to focus on the positions that require actual computation.

## Results

### Single-Step Accuracy (3,682 held-out states after filtering halts)

| Instructions | Full accuracy | Changed-position accuracy |
|---|---|---|
| 1-8 (all) | **100.0%** | **100.0%** |

Zero errors across all instruction counts. The model has learned the complete SUBLEQ algorithm — subtract, compare, branch — with perfect accuracy.

### Multi-Step Program Execution

Programs marked with † were **never in training data in any form**. Multiply single-step transitions appear in training traces, but the model never sees a full multiply program executed.

| Program | Cases | Max steps | Accuracy | Notes |
|---------|-------|-----------|----------|-------|
| Negate | 201 | 3 | 100% | |
| Addition | 300 | 4 | 100% | |
| Countdown (1–20) | 20 | 39 | 100% | |
| Multiply | 141 | ~33 | **100%** | Single steps in training; full programs never seen |
| Fibonacci† | 6 | 47 | **100%** | |
| Division† | 8 | 71 | **87.5%** | div(126,7) fails (gets stuck at 200-step limit) |
| Square root† | 10 | 61 | **100%** | |
| Random programs | 100 | 30 | **97%** | |
| **Total** | **786** | | **99.5%** | |

Longest correct computation: **isqrt(100) = 10**, requiring 61 consecutive correct steps with zero errors.

### Scaling: Width > Depth

| Model | Params | d_model | Layers | Full acc (5K steps) |
|-------|--------|---------|--------|---------------------|
| Tiny | 135K | 64 | 2 | 23.3% |
| Small | 235K | 64 | 4 | 20.6% |
| Medium | 864K | 128 | 4 | 49.1% |
| Base | 1.3M | 128 | 6 | 36.2% |
| **Wide** | **4.9M** | **256** | **6** | **95.9%** |
| Deep | 2.4M | 128 | 12 | 74.8% |

The wide model outperforms the deep model (2× layers, half the params) by 21 percentage points. **Information routing bandwidth** (d_head = d_model / n_heads) is the bottleneck, not depth.

## Usage

```bash
# Train from scratch
python train.py

# Evaluate
python eval.py

# Watch demos (Fibonacci, multiplication, division, square root)
python demo.py

# Interactive REPL — step through programs yourself
python play.py

# Or use the Makefile
make train          # Full training (80K steps)
make train-fast     # Smaller model (d=128), faster (~1 hour)
make demo           # Run demos
make play           # Interactive mode
```

## Files

```
subleq/              # Python package
  interpreter.py     # SUBLEQ interpreter (step, run, clamp)
  tokenizer.py       # Byte-level encode/decode
  programs.py        # Program generators (fib, mul, div, isqrt, ...)
  data.py            # Training data generation
  model.py           # Transformer architecture (MiniSUBLEQTransformer)

train.py             # Training script (one command, auto-detects GPU)
eval.py              # Evaluation (single-step + multi-step)
demo.py              # Impressive demos with ANSI colors
play.py              # Interactive REPL with memory grid display
figures/             # Training curves, rollout visualizations
```
