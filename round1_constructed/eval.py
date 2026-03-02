#!/usr/bin/env python3
"""
Evaluate the hand-coded SUBLEQ transformer against the ground-truth interpreter.

Tests:
  Tier 1: Negate (201 values from -100 to 100)
  Tier 2: Addition (441 pairs, a,b in [-10, 10])
  Tier 3: Multiply (10 pairs)
  Tier 4: Random single-step (1200 random states)
  Tier 5: Random multi-step programs (200 random programs, up to 200 steps)
  Tier 6: Bubble sort (n=2..8, 5 random arrays each = 35 tests)
"""

import sys
import os
import random
import torch

sys.path.insert(0, os.path.dirname(__file__))

from model import HandCodedSUBLEQ
from interpreter import step, run, MEM_SIZE, VALUE_OFFSET, VALUE_MIN, VALUE_MAX, DATA_START
from programs import make_negate, make_addition, make_multiply, make_random_program, make_bubble_sort


def model_step(model, mem, pc):
    """One forward pass of the model."""
    tokens = [pc + VALUE_OFFSET] + [v + VALUE_OFFSET for v in mem]
    inp = torch.tensor([tokens], dtype=torch.long)
    pred = model.predict_step(inp)[0].tolist()
    new_pc = pred[0] - VALUE_OFFSET
    new_mem = [t - VALUE_OFFSET for t in pred[1:]]
    return new_mem, new_pc


def model_run(model, mem, pc, max_steps=5000):
    """Run the model autoregressively until halt or max_steps."""
    m, p = list(mem), pc
    for s in range(max_steps):
        if p < 0 or p + 2 >= MEM_SIZE:
            return m, p, s
        m, p = model_step(model, m, p)
    return m, p, max_steps


def main():
    model = HandCodedSUBLEQ()
    print(f"Hand-coded SUBLEQ Transformer: {model.count_params():,} parameters")
    print(f"Architecture: 4 layers, 8 heads, d_model=32, d_ff=64")
    print(f"Memory: {MEM_SIZE} cells, 16-bit integers [{VALUE_MIN}, {VALUE_MAX}]")
    print()

    total_pass = 0
    total_tests = 0

    # ── Tier 1: Negate ──
    print("Tier 1: Negate (v in [-100, 100])")
    tier_pass = 0
    for v in range(-100, 101):
        mem, pc, r = make_negate(v)
        gt_mem, gt_pc, _ = run(mem, pc, max_steps=200)
        md_mem, md_pc, _ = model_run(model, mem, pc, max_steps=200)
        if md_mem == gt_mem and md_pc == gt_pc:
            tier_pass += 1
    print(f"  {tier_pass}/201 passed")
    total_pass += tier_pass
    total_tests += 201

    # ── Tier 2: Addition ──
    print("Tier 2: Addition (a, b in [-10, 10])")
    tier_pass = 0
    count = 0
    for a in range(-10, 11):
        for b in range(-10, 11):
            count += 1
            mem, pc, r = make_addition(a, b)
            gt_mem, gt_pc, _ = run(mem, pc, max_steps=500)
            md_mem, md_pc, _ = model_run(model, mem, pc, max_steps=500)
            if md_mem == gt_mem and md_pc == gt_pc:
                tier_pass += 1
    print(f"  {tier_pass}/{count} passed")
    total_pass += tier_pass
    total_tests += count

    # ── Tier 3: Multiply ──
    print("Tier 3: Multiply (10 pairs)")
    tier_pass = 0
    pairs = [(2,3), (5,4), (7,1), (0,5), (1,1), (3,7), (6,6), (8,2), (4,9), (10,3)]
    for a, b in pairs:
        mem, pc, r = make_multiply(a, b)
        gt_mem, gt_pc, gt_s = run(mem, pc, max_steps=2000)
        md_mem, md_pc, md_s = model_run(model, mem, pc, max_steps=2000)
        ok = md_mem == gt_mem and md_pc == gt_pc
        if ok:
            tier_pass += 1
        else:
            print(f"  FAIL: {a}*{b} = {gt_mem[r]}, model got {md_mem[r]}")
    print(f"  {tier_pass}/{len(pairs)} passed")
    total_pass += tier_pass
    total_tests += len(pairs)

    # ── Tier 4: Random single-step ──
    print("Tier 4: Random single-step (1200 tests)")
    tier_pass = 0
    random.seed(12345)
    for trial in range(1200):
        mem = [0] * MEM_SIZE
        pc = random.choice([0, 3, 6, 9, 12, 15, 18, 21])
        mem[pc] = random.randint(0, MEM_SIZE - 1)
        mem[pc + 1] = random.randint(0, MEM_SIZE - 1)
        mem[pc + 2] = random.randint(-1, 60)
        for i in range(DATA_START, MEM_SIZE):
            mem[i] = random.randint(-500, 500)
        for i in range(24, 80):
            if i < pc or i > pc + 2:
                mem[i] = random.randint(-300, 300)

        gt_mem, gt_pc, _ = step(list(mem), pc)
        md_mem, md_pc = model_step(model, mem, pc)

        if md_mem == gt_mem and md_pc == gt_pc:
            tier_pass += 1
    print(f"  {tier_pass}/1200 passed")
    total_pass += tier_pass
    total_tests += 1200

    # ── Tier 5: Random multi-step programs ──
    print("Tier 5: Random multi-step programs (200 tests, up to 200 steps)")
    tier_pass = 0
    for seed in range(200):
        mem, pc = make_random_program(seed=seed + 5000)
        gt_mem, gt_pc, gt_s = run(mem, pc, max_steps=200)
        md_mem, md_pc, md_s = model_run(model, mem, pc, max_steps=200)
        if md_mem == gt_mem and md_pc == gt_pc:
            tier_pass += 1
    print(f"  {tier_pass}/200 passed")
    total_pass += tier_pass
    total_tests += 200

    # ── Tier 6: Bubble sort ──
    print("Tier 6: Bubble sort (n=2..8, 5 random arrays each)")
    tier_pass = 0
    tier_count = 0
    for n in range(2, 9):
        for trial in range(5):
            random.seed(n * 1000 + trial)
            values = [random.randint(-50, 50) for _ in range(n)]
            mem, pc, arr_start, arr_n = make_bubble_sort(values)
            gt_mem, gt_pc, gt_s = run(mem, pc, max_steps=50000)
            md_mem, md_pc, md_s = model_run(model, mem, pc, max_steps=50000)
            tier_count += 1
            if md_mem == gt_mem and md_pc == gt_pc:
                tier_pass += 1
            else:
                # Check if at least the array is sorted correctly
                md_arr = md_mem[arr_start:arr_start + n]
                gt_arr = gt_mem[arr_start:arr_start + n]
                print(f"  FAIL n={n} trial={trial}: expected {gt_arr}, model got {md_arr} (steps: gt={gt_s}, md={md_s})")
    print(f"  {tier_pass}/{tier_count} passed")
    total_pass += tier_pass
    total_tests += tier_count

    # ── Summary ──
    print(f"\nTotal: {total_pass}/{total_tests} ({100*total_pass/total_tests:.1f}%)")
    if total_pass == total_tests:
        print("ALL TESTS PASSED")
    return 0 if total_pass == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
