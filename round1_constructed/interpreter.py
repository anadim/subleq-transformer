"""
SUBLEQ interpreter for the hand-coded transformer (Round 1).

416 memory cells, 16-bit signed integers [-32768, 32767].
384 code cells (128 three-word instructions) + 32 data cells.
"""

MEM_SIZE = 416
CODE_SIZE = 384
DATA_START = 384
VALUE_MIN = -32768
VALUE_MAX = 32767
VALUE_OFFSET = 32768      # token = value + VALUE_OFFSET
VOCAB_SIZE = 65538        # 65536 values + 2 (pad/halt)
SEQ_LEN = 417             # 1 PC + 416 memory cells


def clamp(v):
    """Clamp to 16-bit signed range."""
    return max(VALUE_MIN, min(VALUE_MAX, v))


def step(memory, pc):
    """Execute one SUBLEQ step. Returns (new_memory, new_pc, halted)."""
    mem = list(memory)
    if pc < 0 or pc + 2 >= MEM_SIZE:
        return mem, pc, True

    a, b, c = mem[pc], mem[pc + 1], mem[pc + 2]

    if a < 0 or a >= MEM_SIZE or b < 0 or b >= MEM_SIZE:
        return mem, pc, True

    mem[b] = clamp(mem[b] - mem[a])

    if mem[b] <= 0:
        new_pc = c
    else:
        new_pc = pc + 3

    halted = new_pc < 0 or new_pc + 2 >= MEM_SIZE
    return mem, new_pc, halted


def run(memory, pc, max_steps=10000):
    """Run until halt or max_steps. Returns (memory, pc, steps)."""
    mem = list(memory)
    for s in range(max_steps):
        mem, pc, halted = step(mem, pc)
        if halted:
            return mem, pc, s + 1
    return mem, pc, max_steps
