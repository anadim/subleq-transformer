"""
Byte-tokenized SUBLEQ interpreter.

Instead of one token per memory cell, each value is decomposed into
multiple byte tokens. This allows scaling to arbitrary bit widths
while keeping vocab size small (256 for bytes, 16 for nibbles, etc).

Configuration:
    MEM_SIZE: number of memory cells
    BYTES_PER_VALUE: how many bytes per cell (1=8-bit, 2=16-bit, 4=32-bit)
    VOCAB_SIZE: 256 for byte-level, 16 for nibble-level
"""

MEM_SIZE = 32        # 32 memory cells (matches Manchester Baby)
BYTES_PER_VALUE = 1  # Start with 8-bit (256 values: -128 to 127)
VOCAB_SIZE = 256     # One byte per token
VALUE_MIN = -128
VALUE_MAX = 127
CODE_SIZE = 24       # 8 three-word instructions
DATA_START = 24      # cells 24-31 are data (8 cells)

# Sequence: [PC_byte0, ..., PC_byteN, mem[0]_byte0, ..., mem[31]_byteN]
# For 1 byte per value: SEQ_LEN = 1 + 32 = 33
# For 2 bytes per value: SEQ_LEN = 2 + 32*2 = 66
# For 4 bytes per value: SEQ_LEN = 4 + 32*4 = 132
SEQ_LEN = BYTES_PER_VALUE + MEM_SIZE * BYTES_PER_VALUE


def clamp(v):
    return max(VALUE_MIN, min(VALUE_MAX, v))


def step(memory, pc):
    """One SUBLEQ step. Returns (new_memory, new_pc, halted)."""
    n = len(memory)
    if pc < 0 or pc + 2 >= n:
        return list(memory), pc, True

    a = memory[pc]
    b = memory[pc + 1]
    c = memory[pc + 2]

    if a < 0 or a >= n or b < 0 or b >= n:
        return list(memory), pc, True

    mem = list(memory)
    mem[b] = clamp(mem[b] - mem[a])

    if mem[b] <= 0:
        new_pc = c
    else:
        new_pc = pc + 3

    if new_pc < 0 or new_pc + 2 >= n:
        return mem, new_pc, True

    return mem, new_pc, False


def run(memory, pc=0, max_steps=1000):
    """Run to completion."""
    mem = list(memory)
    steps = 0
    while steps < max_steps:
        if pc < 0 or pc + 2 >= len(mem):
            break
        a, b = mem[pc], mem[pc + 1]
        c = mem[pc + 2]
        if a < 0 or a >= len(mem) or b < 0 or b >= len(mem):
            break
        mem[b] = clamp(mem[b] - mem[a])
        if mem[b] <= 0:
            pc = c
        else:
            pc = pc + 3
        steps += 1
        if pc < 0 or pc + 2 >= len(mem):
            break
    return mem, pc, steps
