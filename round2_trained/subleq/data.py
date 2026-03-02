"""
Training data generation for byte-tokenized SUBLEQ (32 cells, 8-bit).
"""

import random
import torch

from .interpreter import step, MEM_SIZE
from .tokenizer import encode, SEQ_LEN, get_changed_positions
from .programs import (make_negate, make_addition, make_countdown,
                       make_multiply, generate_random_state,
                       generate_random_program)

CHANGE_WEIGHT = 100.0


def generate_step_pair(mem, pc):
    """One (input, output, changed_positions) pair from a SUBLEQ step."""
    new_mem, new_pc, halted = step(mem, pc)
    inp = encode(mem, pc)
    out = encode(new_mem, new_pc)
    changed = get_changed_positions(mem, pc)
    return inp, out, changed


def generate_trace_pairs(memory, pc, max_steps=50):
    """Generate all step pairs from running a program to completion."""
    pairs = []
    mem = list(memory)
    for _ in range(max_steps):
        if pc < 0 or pc + 2 >= len(mem):
            break
        a, b = mem[pc], mem[pc + 1]
        if a < 0 or a >= len(mem) or b < 0 or b >= len(mem):
            break
        pair = generate_step_pair(mem, pc)
        pairs.append(pair)
        new_mem, new_pc, halted = step(mem, pc)
        mem, pc = new_mem, new_pc
        if halted:
            break
    return pairs


def generate_batch(batch_size, instr_range=(1, 8), change_weight=CHANGE_WEIGHT):
    """Generate a batch of random single-step training pairs."""
    inputs, outputs, masks = [], [], []

    for _ in range(batch_size):
        n_instr = random.randint(*instr_range)
        mem, pc = generate_random_state(n_instr)
        inp, out, changed = generate_step_pair(mem, pc)
        inputs.append(inp)
        outputs.append(out)
        mask = torch.ones(SEQ_LEN, dtype=torch.float32)
        for pos in changed:
            mask[pos] = change_weight
        masks.append(mask)

    return torch.stack(inputs), torch.stack(outputs), torch.stack(masks)


def generate_trace_batch(count, instr_range=(1, 8), change_weight=CHANGE_WEIGHT):
    """Generate training pairs from program execution traces."""
    inputs, outputs, masks = [], [], []

    while len(inputs) < count:
        choice = random.random()
        if choice < 0.15:
            v = random.randint(-100, 100)
            mem, pc, _ = make_negate(v)
        elif choice < 0.30:
            a, b = random.randint(-60, 60), random.randint(-60, 60)
            mem, pc, _ = make_addition(a, b)
        elif choice < 0.45:
            n = random.randint(1, 20)
            mem, pc, _ = make_countdown(n)
        elif choice < 0.55:
            a = random.randint(1, 10)
            b = random.randint(1, min(12, 127 // max(a, 1)))
            mem, pc, _ = make_multiply(a, b)
        else:
            n_instr = random.randint(*instr_range)
            mem, pc = generate_random_program(n_instr)

        pairs = generate_trace_pairs(mem, pc, max_steps=40)
        for inp, out, changed in pairs:
            if len(inputs) >= count:
                break
            inputs.append(inp)
            outputs.append(out)
            mask = torch.ones(SEQ_LEN, dtype=torch.float32)
            for pos in changed:
                mask[pos] = change_weight
            masks.append(mask)

    return (torch.stack(inputs[:count]),
            torch.stack(outputs[:count]),
            torch.stack(masks[:count]))


def pregenerate_data(size, instr_range=(1, 8)):
    """Pre-generate a dataset. Mix: 60% random states, 40% traces."""
    random_count = int(size * 0.6)
    trace_count = size - random_count

    all_inp, all_out, all_mask = [], [], []

    bs = 512
    for start in range(0, random_count, bs):
        n = min(bs, random_count - start)
        inp, out, mask = generate_batch(n, instr_range=instr_range)
        all_inp.append(inp)
        all_out.append(out)
        all_mask.append(mask)

    if trace_count > 0:
        t_inp, t_out, t_mask = generate_trace_batch(trace_count,
                                                      instr_range=instr_range)
        all_inp.append(t_inp)
        all_out.append(t_out)
        all_mask.append(t_mask)

    inp = torch.cat(all_inp)
    out = torch.cat(all_out)
    mask = torch.cat(all_mask)

    perm = torch.randperm(inp.size(0))
    return inp[perm], out[perm], mask[perm]
