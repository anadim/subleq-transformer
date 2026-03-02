"""
Byte-level tokenizer for SUBLEQ.

Each value is split into BYTES_PER_VALUE bytes using two's complement.
For 8-bit (1 byte): value in [-128, 127] -> unsigned byte [0, 255]
For 16-bit (2 bytes): value in [-32768, 32767] -> 2 bytes big-endian
For 32-bit (4 bytes): value in [-2^31, 2^31-1] -> 4 bytes big-endian

Token layout: [PC_bytes..., mem[0]_bytes..., mem[1]_bytes..., ..., mem[N-1]_bytes...]
"""

import torch
from .interpreter import MEM_SIZE, BYTES_PER_VALUE, VOCAB_SIZE, VALUE_MIN, VALUE_MAX, SEQ_LEN


def value_to_bytes(v, n_bytes=BYTES_PER_VALUE):
    """Convert a signed integer to unsigned byte tokens (big-endian two's complement)."""
    if n_bytes == 1:
        # 8-bit: [-128, 127] -> [0, 255]
        return [v & 0xFF]
    elif n_bytes == 2:
        # 16-bit two's complement, big-endian
        if v < 0:
            v = v + (1 << 16)
        return [(v >> 8) & 0xFF, v & 0xFF]
    elif n_bytes == 4:
        # 32-bit two's complement, big-endian
        if v < 0:
            v = v + (1 << 32)
        return [(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF]
    else:
        raise ValueError(f"Unsupported n_bytes={n_bytes}")


def bytes_to_value(byte_tokens, n_bytes=BYTES_PER_VALUE):
    """Convert unsigned byte tokens back to signed integer."""
    if n_bytes == 1:
        v = byte_tokens[0]
        if v >= 128:
            v -= 256
        return v
    elif n_bytes == 2:
        v = (byte_tokens[0] << 8) | byte_tokens[1]
        if v >= (1 << 15):
            v -= (1 << 16)
        return v
    elif n_bytes == 4:
        v = (byte_tokens[0] << 24) | (byte_tokens[1] << 16) | (byte_tokens[2] << 8) | byte_tokens[3]
        if v >= (1 << 31):
            v -= (1 << 32)
        return v
    else:
        raise ValueError(f"Unsupported n_bytes={n_bytes}")


def encode(memory, pc):
    """Encode (memory, pc) -> LongTensor of shape (SEQ_LEN,)."""
    tokens = value_to_bytes(pc)
    for i in range(MEM_SIZE):
        v = memory[i] if i < len(memory) else 0
        tokens.extend(value_to_bytes(v))
    assert len(tokens) == SEQ_LEN, f"Expected {SEQ_LEN} tokens, got {len(tokens)}"
    return torch.tensor(tokens, dtype=torch.long)


def decode(tokens):
    """Decode LongTensor -> (memory, pc)."""
    if tokens.dim() == 2:
        tokens = tokens[0]
    t = tokens.tolist()

    B = BYTES_PER_VALUE
    pc = bytes_to_value(t[:B])
    memory = []
    for i in range(MEM_SIZE):
        start = B + i * B
        memory.append(bytes_to_value(t[start:start + B]))
    return memory, pc


def get_changed_positions(memory, pc):
    """Token positions that change in one SUBLEQ step.

    Returns list of token indices (not memory cell indices).
    PC always changes -> positions 0..BYTES_PER_VALUE-1.
    mem[b] changes -> positions for cell b.
    """
    B = BYTES_PER_VALUE
    positions = list(range(B))  # PC bytes always change

    n = len(memory)
    if 0 <= pc and pc + 2 < n:
        b = memory[pc + 1]
        if 0 <= b < n:
            start = B + b * B
            positions.extend(range(start, start + B))
    return positions
