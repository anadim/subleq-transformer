"""
Standard Transformer that exactly executes SUBLEQ via hand-coded weights.

Reconstructed from the report: "A Looped Standard Transformer That Is a SUBLEQ
Computer: Constructive Proof via Analytically Hand-Coded Weights."

Architecture: 4 layers, 8 heads, d_model=32, d_head=4, d_ff=64, ReLU, no LayerNorm.
416 memory cells, 16-bit signed integers, 2.1M parameters (~100 nonzero in transformer logic).

Every weight is set analytically. No training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .interpreter import (
        MEM_SIZE, VALUE_MIN, VALUE_MAX, VALUE_OFFSET, VOCAB_SIZE, SEQ_LEN
    )
except ImportError:
    from interpreter import (
        MEM_SIZE, VALUE_MIN, VALUE_MAX, VALUE_OFFSET, VOCAB_SIZE, SEQ_LEN
    )

# ── Architecture constants ──────────────────────────────────────────
D_MODEL = 32
N_HEADS = 8
D_HEAD = 4       # D_MODEL // N_HEADS
D_FF = 64
N_LAYERS = 4

# ── Scale constants (Appendix B) ────────────────────────────────────
SCALE = 10000.0    # Attention sharpness (score gap = 5000 per position)
HUGE = 40000.0     # ReLU safe-gating (> max |v| = 32767, float32-exact)
MUX_M = 70000.0    # PC branch MUX (> max |pc+3-c| ~ 65538)

# ── Dimension layout (Appendix A) ───────────────────────────────────
DV     = 0   # Token value; also receives PC delta and write delta from L4
DI     = 2   # Position index i
DI2    = 3   # Position index squared i^2
D1     = 4   # Constant 1
DPC    = 5   # PC indicator: 1 at position 0 only

DA     = 6   # Operand a = mem[pc]               (L1 attn)
DB     = 8   # Operand b = mem[pc+1]              (L1 attn)
DC     = 10  # Operand c = mem[pc+2]              (L1 attn)
DPCC   = 18  # Broadcast copy of PC value         (L1 attn)
DSB    = 21  # Safe b: b at pos 0, 0 elsewhere    (L1 FFN)

DMA    = 12  # mem[a]                             (L2 attn)
DMB    = 13  # mem[b]                             (L2 attn)
DNV    = 14  # New value = mem[b] - mem[a]        (L2 FFN)
DDW    = 15  # Write delta = -mem[a]              (L2 FFN)
DSTEP  = 20  # Step indicator 1[nv>0] (unsafe)    (L2 FFN)

DSDDW  = 22  # Safe write delta                   (L2 FFN)
DSS    = 29  # Safe step                          (L2 FFN)

DBCB   = 24  # Broadcast of b to all positions    (L3 attn)
DBCDDW = 25  # Broadcast of write delta           (L3 attn)
DH0    = 26  # ReLU(j - b)                        (L3 FFN)
DH1    = 27  # ReLU(j - b - 1)                    (L3 FFN)
DH2    = 28  # ReLU(j - b - 2)                    (L3 FFN)


# ── Standard transformer building blocks ────────────────────────────

class SelfAttention(nn.Module):
    """Multi-head self-attention with standard dot-product QK^T."""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.W_o(out)


class FFN(nn.Module):
    """Feed-forward: Linear -> ReLU -> Linear, with bias."""

    def __init__(self, d_model=D_MODEL, d_ff=D_FF):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class TransformerBlock(nn.Module):
    """Attention + FFN with residual connections, NO LayerNorm."""

    def __init__(self):
        super().__init__()
        self.attn = SelfAttention()
        self.ffn = FFN()

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class HandCodedSUBLEQ(nn.Module):
    """
    4-layer standard transformer with analytically hand-coded weights
    that exactly executes SUBLEQ.

    Input: (B, 417) integer tokens where token = value + 32768.
           Position 0 = PC, positions 1-416 = memory cells.
    Output: logits (B, 417, VOCAB_SIZE) or predicted tokens (B, 417).
    """

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])

        self._init_all_weights()

    def _init_all_weights(self):
        """Set every weight analytically."""
        # Zero everything first
        for p in self.parameters():
            p.data.zero_()

        self._init_embeddings()
        self._init_layer1()
        self._init_layer2()
        self._init_layer3()
        self._init_layer4()

    # ================================================================
    # Embeddings
    # ================================================================

    def _init_embeddings(self):
        """Token emb: dim0=value. Pos emb: dim2=i, dim3=i^2, dim4=1, dim5=pc_ind."""
        te = torch.zeros(VOCAB_SIZE, D_MODEL)
        for t in range(min(VOCAB_SIZE, VALUE_MAX - VALUE_MIN + 1)):
            v = t - VALUE_OFFSET
            te[t, DV] = float(v)
        self.tok_emb.weight = nn.Parameter(te, requires_grad=False)

        pe = torch.zeros(SEQ_LEN, D_MODEL)
        for i in range(SEQ_LEN):
            pe[i, DI] = float(i)
            pe[i, DI2] = float(i * i)
            pe[i, D1] = 1.0
            if i == 0:
                pe[i, DPC] = 1.0
        self.pos_emb.weight = nn.Parameter(pe, requires_grad=False)

    # ================================================================
    # Helper: set attention head weights
    # ================================================================

    def _set_head_qkv(self, layer_idx, head_idx, Wq_rows, Wk_rows, Wv_rows):
        """Set Q, K, V weight rows for one head.

        Each of Wq_rows, Wk_rows, Wv_rows is a list of d_head lists,
        where each inner list contains (src_dim, weight) pairs.
        """
        attn = self.layers[layer_idx].attn
        d = D_HEAD
        base = head_idx * d

        for row_idx, pairs in enumerate(Wq_rows):
            for src_dim, w in pairs:
                attn.W_q.weight.data[base + row_idx, src_dim] = w

        for row_idx, pairs in enumerate(Wk_rows):
            for src_dim, w in pairs:
                attn.W_k.weight.data[base + row_idx, src_dim] = w

        for row_idx, pairs in enumerate(Wv_rows):
            for src_dim, w in pairs:
                attn.W_v.weight.data[base + row_idx, src_dim] = w

    def _set_head_out(self, layer_idx, head_idx, out_entries):
        """Set output projection for one head.

        out_entries: list of (dst_dim, head_row, weight) triples.
        """
        attn = self.layers[layer_idx].attn
        d = D_HEAD
        base = head_idx * d
        for dst_dim, head_row, w in out_entries:
            attn.W_o.weight.data[dst_dim, base + head_row] = w

    # ================================================================
    # Content-based addressing (Trick 1)
    # ================================================================

    def _make_fetch_head(self, layer_idx, head_idx, addr_dim, offset, out_dim):
        """Create a head that reads value from position (addr + offset).

        Uses Gaussian addressing: Q=[1, addr, 0, 0], K=[-s, 2s*k, 0, 0]
        so score = -s*(k - (addr+offset))^2 + const.

        The value dimension reads DV and writes to out_dim.
        """
        s = SCALE

        # Q: row0 = 1 (for the constant/quadratic term)
        #    row1 = addr (the target address)
        #    row2, row3 = 0
        # But we need score = -s*(k - t)^2 where t = addr + offset
        # Expanding: -s*k^2 + 2s*k*t - s*t^2
        # Q dot K = Q[0]*K[0] + Q[1]*K[1] + Q[2]*K[2] + Q[3]*K[3]
        #
        # We want Q dot K_at_pos_k = -s*(k - t)^2 + const
        # = -s*k^2 + 2s*k*t - s*t^2
        #
        # K encodes position k: K = [-s*k^2, 2s*k, 0, 0]  (using DI2 and DI)
        # Q encodes target t:   Q = [1, t, 0, 0]
        # Q dot K = -s*k^2 + 2s*k*t  (the -s*t^2 is constant over k, absorbed by softmax)
        #
        # t = addr + offset, so Q[1] needs addr_dim + offset*D1
        # K[0] = -s * DI2, K[1] = 2s * DI

        Wq = [
            [(D1, 1.0)],                                    # row0: constant 1
            [(addr_dim, 1.0), (D1, float(offset))],         # row1: addr + offset
            [],                                              # row2: unused
            [],                                              # row3: unused
        ]
        Wk = [
            [(DI2, -s)],          # row0: -s * i^2
            [(DI, 2.0 * s)],     # row1: 2s * i
            [],                   # row2: unused
            [],                   # row3: unused
        ]
        Wv = [
            [(DV, 1.0)],         # row0: read value dimension
            [],
            [],
            [],
        ]

        self._set_head_qkv(layer_idx, head_idx, Wq, Wk, Wv)
        self._set_head_out(layer_idx, head_idx, [(out_dim, 0, 1.0)])

    def _make_broadcast_head(self, layer_idx, head_idx, src_dim, dst_dim):
        """Create a head where every position attends to position 0.

        Uses DPC (which is 1 at pos 0, 0 elsewhere) as the key signal.
        Q = [HUGE, 0, 0, 0] (everyone has same large query)
        K = [DPC, 0, 0, 0] (only pos 0 has nonzero key)
        V = [src_dim, 0, 0, 0]
        """
        Wq = [
            [(D1, HUGE)],       # row0: large constant
            [],
            [],
            [],
        ]
        Wk = [
            [(DPC, 1.0)],      # row0: 1 at pos 0, 0 elsewhere
            [],
            [],
            [],
        ]
        Wv = [
            [(src_dim, 1.0)],  # row0: source dimension
            [],
            [],
            [],
        ]

        self._set_head_qkv(layer_idx, head_idx, Wq, Wk, Wv)
        self._set_head_out(layer_idx, head_idx, [(dst_dim, 0, 1.0)])

    # ================================================================
    # Layer 1: "Read the Instruction"
    # ================================================================

    def _init_layer1(self):
        """Layer 1: fetch a, b, c from mem[pc], mem[pc+1], mem[pc+2].

        Head 0: read a = mem[pc]     -> DA   (target = pc + 1 in token space)
        Head 1: read b = mem[pc+1]   -> DB   (target = pc + 2)
        Head 2: read c = mem[pc+2]   -> DC   (target = pc + 3)
        Head 3: broadcast PC to all  -> DPCC

        FFN: compute safe_b (DSB) = b at pos 0, 0 elsewhere.
        """
        # PC is at position 0, mem[k] is at position k+1.
        # So mem[pc] is at position (pc + 1).
        # The address (pc) is in DV at position 0. But we need it available
        # as a dimension that the fetch head can read. The PC value IS in DV
        # at position 0. But every position has DV = its own value.
        #
        # For Layer 1 fetch heads, we use DV as the address dimension.
        # At position 0, DV = pc. The query is computed at position 0.
        # But attention computes Q at ALL positions, so non-pos-0 queries
        # will also try to fetch. That's OK — we only care about pos 0's result.
        #
        # Wait: attention writes to ALL positions. So the fetched value goes
        # to DA/DB/DC at every position, not just pos 0. This is the
        # "contamination" issue — the FFN handles it via safe gating.

        # Head 0: fetch a. Target = DV + 1 (pc + 1 in token layout).
        self._make_fetch_head(0, 0, addr_dim=DV, offset=1, out_dim=DA)

        # Head 1: fetch b. Target = DV + 2.
        self._make_fetch_head(0, 1, addr_dim=DV, offset=2, out_dim=DB)

        # Head 2: fetch c. Target = DV + 3.
        self._make_fetch_head(0, 2, addr_dim=DV, offset=3, out_dim=DC)

        # Head 3: broadcast PC value to all positions.
        self._make_broadcast_head(0, 3, src_dim=DV, dst_dim=DPCC)

        # FFN: safe_b = b at pos 0, 0 elsewhere.
        # Pattern: ReLU(b + H*pc_ind - H) - ReLU(-b + H*pc_ind - H)
        # At pos 0 (pc_ind=1): ReLU(b) - ReLU(-b) = b
        # At other pos (pc_ind=0): ReLU(b-H) - ReLU(-b-H) = 0 (both < 0)
        ffn = self.layers[0].ffn
        H = HUGE

        # h0 = ReLU(DB + H*DPC - H)
        ffn.w1.weight.data[0, DB] = 1.0
        ffn.w1.weight.data[0, DPC] = H
        ffn.w1.bias.data[0] = -H

        # h1 = ReLU(-DB + H*DPC - H)
        ffn.w1.weight.data[1, DB] = -1.0
        ffn.w1.weight.data[1, DPC] = H
        ffn.w1.bias.data[1] = -H

        # DSB = h0 - h1
        ffn.w2.weight.data[DSB, 0] = 1.0
        ffn.w2.weight.data[DSB, 1] = -1.0

    # ================================================================
    # Layer 2: "Fetch the Data and Compute"
    # ================================================================

    def _init_layer2(self):
        """Layer 2: fetch mem[a] and mem[b], compute subtraction and branch.

        Head 0: read mem[a] -> DMA  (target = DA + 1)
        Head 1: read mem[b] -> DMB  (target = DB + 1)

        FFN computes:
          DNV   = mem[b] - mem[a]           (new value)
          DDW   = -mem[a]                   (write delta)
          DSTEP = 1[nv > 0]                 (step indicator, unsafe)
          DSDDW = safe write delta          (gated)
          DSS   = safe step                 (gated)
        """
        # Head 0: fetch mem[a]. Address is DA (set by L1), target = DA + 1.
        self._make_fetch_head(1, 0, addr_dim=DA, offset=1, out_dim=DMA)

        # Head 1: fetch mem[b]. Address is DB, target = DB + 1.
        # But DB is contaminated at non-pos-0! We should use DSB (safe b).
        self._make_fetch_head(1, 1, addr_dim=DSB, offset=1, out_dim=DMB)

        # FFN: 9 hidden units
        ffn = self.layers[1].ffn
        H = HUGE

        # --- h0: new_val = mem[b] - mem[a] ---
        # DNV = DMB - DMA
        ffn.w1.weight.data[0, DMB] = 1.0
        ffn.w1.weight.data[0, DMA] = -1.0
        ffn.w1.bias.data[0] = 0.0
        # But ReLU clips negative! We need the actual value.
        # Use h0 = ReLU(DMB - DMA + H*D1) and h1 = ReLU(-DMB + DMA + H*D1)
        # Then DNV = h0 - h1 - ... no, let's use the standard pattern.
        # Actually: we can compute DNV = DMB - DMA directly in w2 if we route
        # DMB and DMA through separate ReLU(x+H) - ReLU(-x+H) = x gates.
        # But simpler: just output DNV via h0-h1 pattern.
        ffn.w1.weight.data[0, DMB] = 1.0
        ffn.w1.weight.data[0, DMA] = -1.0
        ffn.w1.weight.data[0, D1] = H
        ffn.w1.bias.data[0] = 0.0
        # h0 = ReLU(DMB - DMA + H) >= 0 always (since H > max value range)

        ffn.w1.weight.data[1, DMB] = -1.0
        ffn.w1.weight.data[1, DMA] = 1.0
        ffn.w1.weight.data[1, D1] = H
        ffn.w1.bias.data[1] = 0.0
        # h1 = ReLU(-DMB + DMA + H) >= 0 always

        # DNV = h0 - h1 = (DMB - DMA + H) - (-DMB + DMA + H) = 2*(DMB - DMA)
        # So we need factor 0.5
        ffn.w2.weight.data[DNV, 0] = 0.5
        ffn.w2.weight.data[DNV, 1] = -0.5

        # --- h2, h3: step indicator 1[nv > 0] ---
        # From Trick 2: 1[x>0] = ReLU(x) - ReLU(x-1) for integers.
        # x = new_val = DMB - DMA (at pos 0; garbage elsewhere).
        # h2 = ReLU(DMB - DMA), h3 = ReLU(DMB - DMA - 1)
        ffn.w1.weight.data[2, DMB] = 1.0
        ffn.w1.weight.data[2, DMA] = -1.0
        ffn.w1.bias.data[2] = 0.0

        ffn.w1.weight.data[3, DMB] = 1.0
        ffn.w1.weight.data[3, DMA] = -1.0
        ffn.w1.bias.data[3] = -1.0

        # DSTEP = h2 - h3 (unsafe: contaminated at non-pos-0)
        ffn.w2.weight.data[DSTEP, 2] = 1.0
        ffn.w2.weight.data[DSTEP, 3] = -1.0

        # --- h4: write delta = -mem[a] ---
        # DDW = -DMA. Use: h4 = ReLU(DMA + H), h5 = ReLU(-DMA + H)
        # DDW = -(h4 - h5)/2 = (h5 - h4)/2 ... wait, let's be careful.
        # DDW should be -mem[a] = -DMA.
        # h4 = ReLU(DMA + H) = DMA + H (always positive)
        # h5 = ReLU(-DMA + H) = -DMA + H (always positive)
        # h5 - h4 = -2*DMA, so DDW = (h5 - h4)/2 = -DMA. ✓
        ffn.w1.weight.data[4, DMA] = 1.0
        ffn.w1.weight.data[4, D1] = H
        ffn.w1.bias.data[4] = 0.0

        ffn.w1.weight.data[5, DMA] = -1.0
        ffn.w1.weight.data[5, D1] = H
        ffn.w1.bias.data[5] = 0.0

        ffn.w2.weight.data[DDW, 4] = -0.5
        ffn.w2.weight.data[DDW, 5] = 0.5

        # --- h6, h7: safe write delta (gated by DPC) ---
        # DSDDW = -mem[a] at pos 0, 0 elsewhere
        # h6 = ReLU(DMA + H*DPC - H), h7 = ReLU(-DMA + H*DPC - H)
        # DSDDW = (h7 - h6) / 2  [same sign trick as DDW but gated]
        # Wait: we need DSDDW = -DMA at pos 0.
        # h6 = ReLU(DMA + H*DPC - H). At pos 0: ReLU(DMA) (could be negative!).
        # Hmm, need to be more careful. Use the full safe extraction pattern:
        # h6 = ReLU(DDW + H*DPC - H), h7 = ReLU(-DDW + H*DPC - H)
        # But DDW is computed in this same FFN layer... we can't read it.
        # So we compute DSDDW from DMA directly with the gating:
        # h6 = ReLU(-DMA + H*DPC), h7 = ReLU(DMA + H*DPC)
        # At pos 0: h6 = ReLU(-DMA + H) = -DMA + H, h7 = ReLU(DMA + H) = DMA + H
        # DSDDW = (h6 - h7)/2 = (-DMA + H - DMA - H)/2 = -DMA ✓
        # At other pos: h6 = ReLU(-DMA), h7 = ReLU(DMA)
        # If DMA > 0: h6=0, h7=DMA, (h6-h7)/2 = -DMA/2 ← WRONG, not zero!
        # We need the full gating with -H bias:
        # h6 = ReLU(-DMA + H*DPC - H), h7 = ReLU(DMA + H*DPC - H)
        # At pos 0 (DPC=1): h6=ReLU(-DMA), h7=ReLU(DMA). DSDDW=(h6-h7)/2
        #   If DMA>0: h6=0, h7=DMA → -DMA/2. But we want -DMA. ← Still wrong factor.
        #
        # The correct safe extraction pattern from the report:
        # safe_x = ReLU(x + H*pc_ind - H) - ReLU(-x + H*pc_ind - H)
        # At pos 0: ReLU(x) - ReLU(-x) = x (works for any x since |x| < H)
        #   Wait: ReLU(x) - ReLU(-x) = x for all x? Let's check:
        #   x>0: ReLU(x) - ReLU(-x) = x - 0 = x ✓
        #   x<0: ReLU(x) - ReLU(-x) = 0 - (-x) = x ✓
        #   x=0: 0 - 0 = 0 ✓. YES, ReLU(x) - ReLU(-x) = x always!
        # At other pos: ReLU(x-H) - ReLU(-x-H) = 0 - 0 = 0 ✓ (since |x| < H)
        #
        # So for DSDDW = -DMA at pos 0, 0 elsewhere:
        # h6 = ReLU(-DMA + H*DPC - H), h7 = ReLU(DMA + H*DPC - H)
        # DSDDW = h6 - h7 = safe(-DMA) = -DMA at pos 0, 0 elsewhere ✓

        ffn.w1.weight.data[6, DMA] = -1.0
        ffn.w1.weight.data[6, DPC] = H
        ffn.w1.bias.data[6] = -H

        ffn.w1.weight.data[7, DMA] = 1.0
        ffn.w1.weight.data[7, DPC] = H
        ffn.w1.bias.data[7] = -H

        ffn.w2.weight.data[DSDDW, 6] = 1.0
        ffn.w2.weight.data[DSDDW, 7] = -1.0

        # --- h8, h9 (units 8,9): safe step (gated) ---
        # DSS = step at pos 0, 0 elsewhere
        # step = 1[nv > 0] = ReLU(DMB-DMA) - ReLU(DMB-DMA-1)
        # But step is in [0,1], and we need safe extraction.
        # Directly: compute safe step from scratch using DPC gating.
        # h8 = ReLU(DMB - DMA + H*DPC - H)  [nv gated]
        # h9 = ReLU(DMB - DMA - 1 + H*DPC - H) [nv-1 gated]
        # DSS = h8 - h9
        # At pos 0 (DPC=1): h8=ReLU(DMB-DMA), h9=ReLU(DMB-DMA-1) → step ✓
        # At other pos (DPC=0): h8=ReLU(DMB-DMA-H), h9=ReLU(DMB-DMA-1-H) → 0 ✓

        ffn.w1.weight.data[8, DMB] = 1.0
        ffn.w1.weight.data[8, DMA] = -1.0
        ffn.w1.weight.data[8, DPC] = H
        ffn.w1.bias.data[8] = -H

        ffn.w1.weight.data[9, DMB] = 1.0
        ffn.w1.weight.data[9, DMA] = -1.0
        ffn.w1.weight.data[9, DPC] = H
        ffn.w1.bias.data[9] = -H - 1.0

        ffn.w2.weight.data[DSS, 8] = 1.0
        ffn.w2.weight.data[DSS, 9] = -1.0

    # ================================================================
    # Layer 3: "Tell Everyone Where to Write"
    # ================================================================

    def _init_layer3(self):
        """Layer 3: broadcast b and write delta to all positions.
        FFN computes hat function components.

        Head 0: broadcast safe_b (DSB) -> DBCB
        Head 1: broadcast safe_ddw (DSDDW) -> DBCDDW

        FFN computes (at every position j):
          DH0 = ReLU(j - b)
          DH1 = ReLU(j - b - 1)
          DH2 = ReLU(j - b - 2)
        """
        # Broadcast heads
        self._make_broadcast_head(2, 0, src_dim=DSB, dst_dim=DBCB)
        self._make_broadcast_head(2, 1, src_dim=DSDDW, dst_dim=DBCDDW)

        # FFN: hat function components
        # j is the position index (DI), b is broadcast (DBCB).
        # But DBCB holds b (the operand address). In token layout,
        # the target memory position is b+1. The hat function needs
        # to fire at position b+1:
        #   DH0 = ReLU(j - (b+1))     = ReLU(DI - DBCB - 1)
        #   DH1 = ReLU(j - (b+1) - 1) = ReLU(DI - DBCB - 2)
        #   DH2 = ReLU(j - (b+1) - 2) = ReLU(DI - DBCB - 3)
        # Wait: the report says DH0 = ReLU(j - b), but b there means
        # the target position in token space, which is operand_b + 1.
        # Let me re-read...
        #
        # Report Section 4.6: "DH0 = ReLU(j - b)"
        # Report Section 4.7: "1[j = b+1] = ReLU(j-b) - 2*ReLU(j-b-1) + ReLU(j-b-2)"
        # And from Trick 4 (Eq 10): "1[j = b+1]"
        #
        # So in the report, "b" in the hat function IS the broadcast operand address.
        # The hat fires at j = b+1 in TOKEN space. But mem[b] lives at
        # token position b+1. So the broadcast value DBCB should be the
        # raw operand b, and the hat fires at position b+1. ✓
        #
        # DH0 = ReLU(j - DBCB)
        # DH1 = ReLU(j - DBCB - 1)
        # DH2 = ReLU(j - DBCB - 2)
        # Indicator = DH0 - 2*DH1 + DH2 = 1 at j=DBCB+1, 0 elsewhere.

        ffn = self.layers[2].ffn

        # h0 = ReLU(DI - DBCB) = ReLU(j - b)
        ffn.w1.weight.data[0, DI] = 1.0
        ffn.w1.weight.data[0, DBCB] = -1.0
        ffn.w1.bias.data[0] = 0.0

        # h1 = ReLU(DI - DBCB - 1)
        ffn.w1.weight.data[1, DI] = 1.0
        ffn.w1.weight.data[1, DBCB] = -1.0
        ffn.w1.bias.data[1] = -1.0

        # h2 = ReLU(DI - DBCB - 2)
        ffn.w1.weight.data[2, DI] = 1.0
        ffn.w1.weight.data[2, DBCB] = -1.0
        ffn.w1.bias.data[2] = -2.0

        # Output to DH0, DH1, DH2
        ffn.w2.weight.data[DH0, 0] = 1.0
        ffn.w2.weight.data[DH1, 1] = 1.0
        ffn.w2.weight.data[DH2, 2] = 1.0

    # ================================================================
    # Layer 4: "Write the Result"
    # ================================================================

    def _init_layer4(self):
        """Layer 4: write delta to mem[b] and compute new PC.

        Attention: inactive (all zeros).

        FFN uses 8 hidden units for two parallel computations:
          Memory write (h0, h1): MUX using hat indicator from L3
          PC update (h2-h7): branch MUX + c extraction + old_pc extraction
        """
        # Layer 4 attention stays zero (already zeroed).

        ffn = self.layers[3].ffn
        M = MUX_M
        H = HUGE

        # ── Memory write: h0, h1 ──
        # Binary MUX (Trick 3): δ * indicator
        # indicator = DH0 - 2*DH1 + DH2 (= 1 at target, 0 elsewhere)
        # δ = DBCDDW (broadcast write delta = -mem[a])
        #
        # h0 = ReLU(δ + M*(DH0 - 2*DH1 + DH2) - M)
        # h1 = ReLU(-δ + M*(DH0 - 2*DH1 + DH2) - M)
        # write_delta_at_j = h0 - h1 = δ at target, 0 elsewhere.

        ffn.w1.weight.data[0, DBCDDW] = 1.0
        ffn.w1.weight.data[0, DH0] = M
        ffn.w1.weight.data[0, DH1] = -2.0 * M
        ffn.w1.weight.data[0, DH2] = M
        ffn.w1.bias.data[0] = -M

        ffn.w1.weight.data[1, DBCDDW] = -1.0
        ffn.w1.weight.data[1, DH0] = M
        ffn.w1.weight.data[1, DH1] = -2.0 * M
        ffn.w1.weight.data[1, DH2] = M
        ffn.w1.bias.data[1] = -M

        # DV += h0 - h1 (memory write delta)
        ffn.w2.weight.data[DV, 0] = 1.0
        ffn.w2.weight.data[DV, 1] = -1.0

        # ── PC update: h2-h7 ──
        # All gated to fire only at position 0.
        #
        # new_pc = step*(pc+3) + (1-step)*c = c + step*(pc+3-c)
        # delta_pc = new_pc - old_pc = (c - old_pc) + step*(pc+3-c)
        #
        # We write delta_pc to DV at pos 0. After residual: DV = old_pc + delta_pc = new_pc.
        #
        # h2, h3: MUX for step*(pc+3-c) using DSS (safe step)
        #   z = pc + 3 - c. We read DPCC (broadcast PC) for pc, DC for c.
        #   But DC is contaminated at non-pos-0. Gate with DPC.
        #   Actually: the MUX already uses DSS which is 0 at non-pos-0.
        #   z = DPCC + 3*D1 - DC
        #   s = DSS
        #   h2 = ReLU(z + 2*M*s - M) = ReLU(DPCC + 3 - DC + 2*M*DSS - M)
        #   h3 = ReLU(-z + 2*M*s - M) = ReLU(-DPCC - 3 + DC + 2*M*DSS - M)
        #   step*z = (h2 - h3) / 2

        Mpc = MUX_M
        ffn.w1.weight.data[2, DPCC] = 1.0
        ffn.w1.weight.data[2, D1] = 3.0
        ffn.w1.weight.data[2, DC] = -1.0
        ffn.w1.weight.data[2, DSS] = 2.0 * Mpc
        ffn.w1.bias.data[2] = -Mpc

        ffn.w1.weight.data[3, DPCC] = -1.0
        ffn.w1.weight.data[3, D1] = -3.0
        ffn.w1.weight.data[3, DC] = 1.0
        ffn.w1.weight.data[3, DSS] = 2.0 * Mpc
        ffn.w1.bias.data[3] = -Mpc

        # h4, h5: extract c (gated by DPC)
        # c_safe = ReLU(DC + H*DPC - H) - ReLU(-DC + H*DPC - H)
        ffn.w1.weight.data[4, DC] = 1.0
        ffn.w1.weight.data[4, DPC] = H
        ffn.w1.bias.data[4] = -H

        ffn.w1.weight.data[5, DC] = -1.0
        ffn.w1.weight.data[5, DPC] = H
        ffn.w1.bias.data[5] = -H

        # h6, h7: extract old_pc (gated by DPC)
        # We need old_pc at pos 0. DV at pos 0 = old_pc (from token embedding).
        # But DV may have been modified by Layer 4 write (h0-h1)... No: h0-h1
        # writes at pos b+1, not pos 0 (unless b+1 = 0, which means b = -1,
        # not a valid address). So DV at pos 0 is still old_pc.
        # Actually wait: DV at pos 0 could have been modified by the memory
        # write in h0-h1 of THIS SAME FFN. But h0-h1 and h6-h7 are all in
        # the same FFN computation. The FFN reads x (the input to this layer),
        # computes all hidden units, then writes. So h6,h7 read from the
        # SAME input x as h0,h1. At pos 0, x[DV] = old_pc. ✓
        #
        # Actually, we should use DPCC (broadcast copy of PC from L1)
        # instead of DV, because DV may have been modified by L2/L3 residuals.
        # L2 attention puts DMA, DMB into their own dims, not DV. ✓
        # L2 FFN writes to DNV, DDW, DSTEP, DSDDW, DSS — not DV. ✓
        # L3 attention writes to DBCB, DBCDDW — not DV. ✓
        # L3 FFN writes to DH0, DH1, DH2 — not DV. ✓
        # So DV at pos 0 IS still old_pc entering Layer 4. But let's use
        # DPCC for safety (it's a clean copy from L1).

        ffn.w1.weight.data[6, DPCC] = 1.0
        ffn.w1.weight.data[6, DPC] = H
        ffn.w1.bias.data[6] = -H

        ffn.w1.weight.data[7, DPCC] = -1.0
        ffn.w1.weight.data[7, DPC] = H
        ffn.w1.bias.data[7] = -H

        # delta_pc = (c - old_pc) + step*(pc+3-c)
        #          = (h4 - h5) - (h6 - h7) + (h2 - h3)/2
        # DV += delta_pc at pos 0 (zero elsewhere due to gating)
        ffn.w2.weight.data[DV, 2] = 0.5    # step*(pc+3-c) / 2
        ffn.w2.weight.data[DV, 3] = -0.5
        ffn.w2.weight.data[DV, 4] = 1.0    # +c
        ffn.w2.weight.data[DV, 5] = -1.0
        ffn.w2.weight.data[DV, 6] = -1.0   # -old_pc
        ffn.w2.weight.data[DV, 7] = 1.0

    # ================================================================
    # Forward / predict interface
    # ================================================================

    def forward(self, tokens):
        """Forward pass returning logits (B, T, VOCAB_SIZE)."""
        B, T = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        for layer in self.layers:
            x = layer(x)

        # Read DV (dimension 0) from every position — uniform extraction.
        values = x[:, :, DV]
        output_tokens = values.round().clamp(VALUE_MIN, VALUE_MAX).long() + VALUE_OFFSET

        # Convert to one-hot-ish logits
        logits = torch.full((B, T, VOCAB_SIZE), -100.0, device=tokens.device)
        logits.scatter_(2, output_tokens.unsqueeze(2), 100.0)
        return logits

    def predict_step(self, tokens):
        """Predict next state tokens directly."""
        squeeze = tokens.dim() == 1
        if squeeze:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        with torch.no_grad():
            x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
            for layer in self.layers:
                x = layer(x)
            values = x[:, :, DV]
            output_tokens = values.round().clamp(VALUE_MIN, VALUE_MAX).long() + VALUE_OFFSET
        if squeeze:
            output_tokens = output_tokens.squeeze(0)
        return output_tokens

    def count_params(self):
        return sum(p.numel() for p in self.parameters()) + sum(b.numel() for b in self.buffers())


if __name__ == '__main__':
    model = HandCodedSUBLEQ()
    print(f"HandCodedSUBLEQ parameters: {model.count_params():,}")
    x = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    pred = model.predict_step(x)
    print(f"predict_step: {x.shape} -> {pred.shape}")
