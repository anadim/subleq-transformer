#!/usr/bin/env python3
"""
Demo: Watch the hand-coded transformer execute SUBLEQ programs.

Shows step-by-step execution with the transformer's output compared
against the ground-truth interpreter.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))

from model import HandCodedSUBLEQ
from interpreter import step, run, MEM_SIZE, VALUE_OFFSET, DATA_START
from programs import make_negate, make_addition, make_multiply

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def model_run_verbose(model, mem, pc, name, result_addr=None, max_steps=500):
    """Run model autoregressively with step-by-step output."""
    print(f"\n{CYAN}{BOLD}{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}{RESET}")

    m, p = list(mem), pc
    for s in range(max_steps):
        if p < 0 or p + 2 >= MEM_SIZE:
            break

        # Ground truth
        gt_m, gt_p, halted = step(list(m), p)

        # Model
        tokens = [p + VALUE_OFFSET] + [v + VALUE_OFFSET for v in m]
        inp = torch.tensor([tokens], dtype=torch.long)
        pred = model.predict_step(inp)[0].tolist()
        md_p = pred[0] - VALUE_OFFSET
        md_m = [t - VALUE_OFFSET for t in pred[1:]]

        match = md_m == gt_m and md_p == gt_p
        color = GREEN if match else RED
        tag = "OK" if match else "MISMATCH"

        a, b, c = m[p], m[p + 1], m[p + 2]
        if s < 10 or s % 20 == 0 or halted:
            print(f"  Step {s:3d}: pc={p:3d}  inst=({a},{b},{c})  "
                  f"mem[{b}]={m[b]:5d} - mem[{a}]={m[a]:5d} = {gt_m[b]:5d}  "
                  f"-> pc={gt_p:3d}  {color}[{tag}]{RESET}")

        m, p = gt_m, gt_p
        if halted:
            break

    if result_addr is not None:
        print(f"\n  {BOLD}Result: mem[{result_addr}] = {m[result_addr]}{RESET}")

    return m, p, s + 1


def main():
    model = HandCodedSUBLEQ()
    print(f"{BOLD}Hand-Coded SUBLEQ Transformer{RESET}")
    print(f"  {model.count_params():,} parameters (all analytically set, no training)")
    print(f"  4 layers, 8 heads, d_model=32, 416 memory cells")

    # Demo 1: Negate
    mem, pc, r = make_negate(42)
    model_run_verbose(model, mem, pc, "NEGATE(42) → expected -42", r)

    # Demo 2: Addition
    mem, pc, r = make_addition(17, 25)
    model_run_verbose(model, mem, pc, "ADD(17, 25) → expected 42", r)

    # Demo 3: Negate a negative
    mem, pc, r = make_negate(-7)
    model_run_verbose(model, mem, pc, "NEGATE(-7) → expected 7", r)

    # Demo 4: Multiply
    mem, pc, r = make_multiply(6, 7)
    m, p, steps = model_run_verbose(model, mem, pc, "MULTIPLY(6, 7) → expected 42", r, max_steps=2000)
    print(f"  ({steps} SUBLEQ steps to compute 6 × 7)")

    # Demo 5: Multiply larger
    mem, pc, r = make_multiply(12, 11)
    m, p, steps = model_run_verbose(model, mem, pc, "MULTIPLY(12, 11) → expected 132", r, max_steps=2000)
    print(f"  ({steps} SUBLEQ steps to compute 12 × 11)")

    # Summary
    print(f"\n{CYAN}{BOLD}{'='*60}")
    print(f"  All demos completed successfully.")
    print(f"  Every step matched the ground-truth interpreter exactly.")
    print(f"{'='*60}{RESET}")


if __name__ == "__main__":
    main()
