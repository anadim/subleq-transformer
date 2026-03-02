# Round 1: Hand-Coded SUBLEQ Transformer

A standard 4-layer transformer with **analytically hand-coded weights** that exactly executes SUBLEQ, a Turing-complete one-instruction computer. No training — every weight is set by mathematical construction.

## Architecture

- **4 layers**, 8 heads, d_model=32, d_head=4, d_ff=64
- **ReLU activation**, no LayerNorm, no causal mask (bidirectional)
- **2,143,712 total parameters** (~2.12M embeddings, ~25K transformer logic, ~100 nonzero)
- **416 memory cells**, 16-bit signed integers [-32768, 32767]

## The 4-Layer Data Flow

| Layer | What it does | Mechanism |
|-------|-------------|-----------|
| 1 | **Read the instruction**: fetch a, b, c from mem[pc], mem[pc+1], mem[pc+2] | Content-based addressing (Gaussian attention peaks) |
| 2 | **Fetch data & compute**: read mem[a], mem[b]; subtract; determine branch | Second pointer dereference + ReLU arithmetic |
| 3 | **Broadcast & build indicator**: tell all positions where to write | Broadcast attention + hat function (3 ReLU ramps) |
| 4 | **Write result & update PC**: apply delta to exactly one cell | Binary MUX (ReLU if-then-else) |

## Usage

```bash
python demo.py    # Watch step-by-step execution
python eval.py    # Full test suite (2,087 tests)
```

## The Four Key Tricks

1. **Content-based addressing**: `q·k = -s(k-t)²` via quadratic identity in d_head=4
2. **Integer step function**: `1[x>0] = ReLU(x) - ReLU(x-1)` (exact for integers)
3. **Binary multiplexer**: `s·z = ½[ReLU(z + 2Ms - M) - ReLU(-z + 2Ms - M)]`
4. **Hat function**: `1[j=b+1] = ReLU(j-b) - 2·ReLU(j-b-1) + ReLU(j-b-2)`

## Why 4 Layers?

The hat function (Trick 4) is one level of ReLU. Multiplying it by the write delta (Trick 3) is a second level. A single FFN computes one level. So we need two FFN layers for selective memory writes — that's Layers 3 and 4.
