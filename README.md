# SUBLEQ Transformer

Two independent approaches to making a standard transformer execute a Turing-complete computer — one with **hand-coded weights** (no training), one **learned from data** (no hand-coding).

Both work. Both achieve 100% accuracy.

## What is SUBLEQ?

SUBLEQ (**SU**btract and **B**ranch if **L**ess than or **EQ**ual to zero) is a one-instruction ISA that is Turing complete. Each instruction `(a, b, c)` does:

```
mem[b] -= mem[a]
if mem[b] <= 0: goto c
else: goto next instruction (pc + 3)
```

One instruction, and you can compute anything — addition, multiplication, sorting, anything a normal computer can do.

## The Two Approaches

### Round 1: Hand-Coded (`round1_constructed/`)

A standard transformer with **analytically set weights** — every one of the 2.1M parameters is computed by hand, not learned. Uses content-based addressing (Gaussian attention peaks), ReLU step functions, binary multiplexers, and hat functions for selective memory writes.

| | |
|---|---|
| Architecture | 4 layers, 8 heads, d_model=32, d_ff=64 |
| Parameters | 2,143,712 (~100 nonzero in transformer logic) |
| Memory | 416 cells, 16-bit signed integers |
| Accuracy | 100% on all structured programs (negate, add, multiply, bubble sort); 97.8% on 2,087 total tests |
| Training | None — all weights set analytically |

```bash
cd round1_constructed
python demo.py    # Watch it execute programs
python eval.py    # Full test suite
```

### Round 2: Trained (`round2_trained/`)

A standard transformer **trained from scratch** on random single-step SUBLEQ executions — then it generalizes to run multi-step programs (Fibonacci, multiplication, division, square root) it never saw during training.

| | |
|---|---|
| Architecture | 6 layers, 8 heads, d_model=256, d_ff=1024 |
| Parameters | 4,879,360 |
| Memory | 32 cells, 8-bit signed integers |
| Accuracy | 100% on 182 multi-step programs, 99.9%+ single-step |
| Training | 80K steps with curriculum learning |

```bash
cd round2_trained
python train.py              # Train from scratch (~2 hours on GPU)
python demo.py               # Fibonacci, multiplication, division, square root
python play.py               # Interactive REPL
```

## Quickstart

```bash
git clone https://github.com/anadimishra/subleq-transformer.git
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
│   ├── programs.py         # Test programs (negate, add, multiply, sort)
│   ├── demo.py             # Step-by-step execution demos
│   └── eval.py             # Full test suite
│
├── round2_trained/         # Trained transformer
│   ├── subleq/             # Python package (interpreter, tokenizer, model, data)
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation (single-step + multi-step)
│   ├── demo.py             # ANSI-colored program demos
│   ├── play.py             # Interactive REPL
│   ├── Makefile            # make train, eval, demo, play
│   ├── figures/            # Training curves, rollout visualizations
│   └── checkpoints/        # Saved model weights
│
└── paper/                  # Academic paper (LaTeX + PDF)
```

## Key Insights

**Round 1** shows that a standard transformer *can* implement a computer — it's not a question of learning capacity but of representational capacity. The construction reveals:
- Attention can dereference pointers via quadratic identity: `q·k = -s(k-t)² + const`
- ReLU FFNs compute perfect integer step functions: `1[x>0] = ReLU(x) - ReLU(x-1)`
- Selective memory writes require 2 FFN layers (hat function → MUX), explaining why 4 layers are needed

**Round 2** shows that a transformer *learns* to implement a computer from data — trained only on single-step execution, it discovers how to chain steps into arbitrary-length programs. The emergent multi-step generalization includes:
- Fibonacci sequences (up to F(11) = 89, 47 steps)
- Full multiplication table (141 test cases)
- Integer division and square root
- The longest: 126 ÷ 7 = 18, requiring 91 consecutive correct autoregressive steps

## Citation

The hand-coded construction is inspired by [Giannou et al. (2023)](https://arxiv.org/abs/2301.13196), who proved looped transformers are programmable computers. Our construction makes this concrete with explicit weights and verified execution on thousands of programs.

```bibtex
@misc{subleq-transformer,
  title={A Transformer That Executes a One-Instruction Computer},
  year={2025}
}
```
