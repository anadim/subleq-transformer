# Round 2: Trained SUBLEQ Transformer

A 4.9M-parameter transformer **trained from scratch** to execute SUBLEQ — then it generalizes to run programs (Fibonacci, multiplication, division, square root) it never saw during training.

## Architecture

- **6 layers**, 8 heads, d_model=256, d_ff=1024
- **Pre-LN**, GELU, bidirectional attention (encoder-only)
- **4,879,360 parameters** (learned via gradient descent)
- **32 memory cells**, 8-bit signed integers [-128, 127]

## Training

- **80K steps** with cosine LR schedule (peak 3e-4) and warmup
- **Curriculum**: gradually increase instruction count (1-2 → 1-4 → 1-6 → 1-8)
- **Data**: generated on-the-fly; 60% random single-step states, 40% program traces
- **Loss**: weighted cross-entropy (100:1 weight on changed positions)
- **Hardware**: ~2 hours on Apple M1/M2 or CUDA GPU

## Results

### Single-step execution
**99.9%+ accuracy** on 2,000 held-out random states across all instruction counts.

### Emergent multi-step programs (never in training data)

| Program | Test cases | Max steps | Accuracy |
|---------|-----------|-----------|----------|
| Fibonacci (F(0) to F(11)) | 5 | 47 | **100%** |
| Multiplication (full 12x12 table) | 141 | 36 | **100%** |
| Division | 16 | 91 | **100%** |
| Square root | 20 | 61 | **100%** |

The longest computation: **126 ÷ 7 = 18**, requiring 91 consecutive correct autoregressive steps.

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
make train          # Full training
make train-fast     # Smaller model, faster (~30 min)
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
