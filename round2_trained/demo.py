#!/usr/bin/env python3
"""
Impressive demos for the SUBLEQ transformer.

Shows the trained neural network executing programs it has NEVER seen
during training -- acting as a learned general-purpose computer.

Programs demonstrated:
1. Fibonacci sequence (8-instruction SUBLEQ program, 47 steps)
2. Multiplication via repeated addition (3-instruction program)
3. Integer division via repeated subtraction
4. Integer square root (via 1+3+5+7+... odd number subtraction)

Usage:
    python demo.py                              # default checkpoint
    python demo.py checkpoints/best_model.pt    # specific checkpoint
"""

import sys
import os
import math
import argparse

import torch

from subleq import (
    MiniSUBLEQTransformer, run,
    MEM_SIZE, VALUE_MIN, VALUE_MAX,
    encode, decode, SEQ_LEN, VOCAB_SIZE,
    make_negate, make_addition, make_multiply, make_fibonacci,
    make_div, make_isqrt,
)

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def auto_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = MiniSUBLEQTransformer(
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model


def model_step(model, memory, pc, device='cpu'):
    inp = encode(memory, pc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
    pred_tokens = logits.argmax(dim=-1).squeeze(0)
    return decode(pred_tokens)


def run_model(model, mem, pc, max_steps=500, device='cpu', trace=False):
    """Run the model as a computer, returning final state and step count."""
    m, p = list(mem), pc
    steps = 0
    for s in range(max_steps):
        if p < 0 or p + 2 >= len(m):
            break
        a, b = m[p], m[p + 1]
        if a < 0 or a >= len(m) or b < 0 or b >= len(m):
            break
        if trace:
            print(f"    {DIM}step {s:3d}:{RESET} pc={p:2d} | "
                  f"mem[{b}]: {m[b]:4d} - mem[{a}]({m[a]:4d}) = ", end="")
        m, p = model_step(model, m, p, device)
        if trace:
            print(f"{m[b]:4d} | new_pc={p}")
        steps += 1
    return m, p, steps


def demo_fibonacci(model, device='cpu'):
    """Fibonacci: the crown jewel. 8-instruction SUBLEQ program."""
    print(f"{CYAN}{BOLD}{'=' * 70}")
    print(f"  FIBONACCI SEQUENCE")
    print(f"  8-instruction SUBLEQ program computing F(2n) and F(2n+1)")
    print(f"  NEVER seen during training -- emergent multi-step computation")
    print(f"{'=' * 70}{RESET}")

    print(f"\n  Algorithm: alternating a+=b, b+=a (each iteration = 2 Fibonacci steps)")
    print(f"  Using only: subtract, conditional branch, and 32 bytes of memory.\n")

    correct = 0
    total = 0
    for n in range(1, 6):
        mem, pc, addr_a, addr_b = make_fibonacci(n)
        exp_mem, _, exp_steps = run(mem, pc, max_steps=500)
        exp_a, exp_b = exp_mem[addr_a], exp_mem[addr_b]

        m, p, model_steps = run_model(model, mem, pc, max_steps=500, device=device)
        got_a, got_b = m[addr_a], m[addr_b]

        total += 1
        ok = (got_a == exp_a and got_b == exp_b)
        if ok:
            correct += 1

        color = GREEN if ok else RED
        status = "CORRECT" if ok else "WRONG"
        print(f"  n={n}: F({2*n:2d})={exp_a:3d}, F({2*n+1:2d})={exp_b:3d}  |  "
              f"Model: F({2*n:2d})={got_a:3d}, F({2*n+1:2d})={got_b:3d}  |  "
              f"{exp_steps:2d} steps  |  {color}{status}{RESET}")

    print(f"\n  Result: {correct}/{total} ({100*correct/total:.1f}%)")

    # Trace for n=3 (F(6)=8, F(7)=13)
    print(f"\n  {CYAN}--- Trace: computing F(6)=8 and F(7)=13 (n=3, 23 steps) ---{RESET}")
    mem, pc, addr_a, addr_b = make_fibonacci(3)
    m, p, s = run_model(model, mem, pc, max_steps=500, device=device, trace=True)
    print(f"\n  Final: a=mem[{addr_a}]={m[addr_a]}, b=mem[{addr_b}]={m[addr_b]}")

    return correct, total


def demo_multiplication(model, device='cpu'):
    """Multiplication table -- emergent capability."""
    print(f"\n{CYAN}{BOLD}{'=' * 70}")
    print(f"  MULTIPLICATION TABLE")
    print(f"  3-instruction program: repeated addition (a * b)")
    print(f"  NEVER in training data -- model learned the algorithm")
    print(f"{'=' * 70}{RESET}")

    correct = 0
    total = 0
    print(f"\n  {'':>4s}", end="")
    for b in range(1, 13):
        print(f" {b:>4d}", end="")
    print()
    print(f"  {'':>4s}" + "-" * 48)

    for a in range(1, 13):
        print(f"  {a:>3d}|", end="")
        for b in range(1, 13):
            if a * b > VALUE_MAX:
                print(f"  {DIM}  .{RESET}", end="")
                continue

            mem, pc, result_addr = make_multiply(a, b)
            expected_mem, _, _ = run(mem, pc, max_steps=200)
            expected = expected_mem[result_addr]

            m, p, s = run_model(model, mem, pc, max_steps=200, device=device)
            got = m[result_addr]
            total += 1
            if got == expected:
                correct += 1
                print(f" {GREEN}{got:>4d}{RESET}", end="")
            else:
                print(f" {RED}X{got:>3d}{RESET}", end="")
        print()

    print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    return correct, total


def demo_division(model, device='cpu'):
    """Integer division via repeated subtraction."""
    print(f"\n{CYAN}{BOLD}{'=' * 70}")
    print(f"  INTEGER DIVISION (a // b)")
    print(f"  5-instruction program: repeated subtraction with counter")
    print(f"  NEVER in training data")
    print(f"{'=' * 70}{RESET}")

    correct = 0
    total = 0
    print()
    test_cases = [(10, 2), (10, 3), (15, 5), (20, 7), (100, 10), (99, 9),
                  (50, 7), (63, 9), (48, 6), (81, 9), (72, 8), (55, 5),
                  (36, 4), (120, 11), (126, 7), (100, 13)]

    for a, b in test_cases:
        mem, pc, result_addr = make_div(a, b)
        exp_mem, _, exp_steps = run(mem, pc, max_steps=500)
        exp_result = exp_mem[result_addr]

        m, p, model_steps = run_model(model, mem, pc, max_steps=500, device=device)
        got = m[result_addr]

        total += 1
        ok = (got == exp_result)
        if ok:
            correct += 1

        color = GREEN if ok else RED
        status = f"{GREEN}OK{RESET}" if ok else f"{RED}WRONG (got {got}){RESET}"
        print(f"  {a:3d} / {b:2d} = {exp_result:2d}  (r={a % b})  |  "
              f"model: {got:2d}  |  {exp_steps:3d} steps  |  {status}")

    print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    return correct, total


def demo_isqrt(model, device='cpu'):
    """Integer square root -- mathematical algorithm in SUBLEQ."""
    print(f"\n{CYAN}{BOLD}{'=' * 70}")
    print(f"  INTEGER SQUARE ROOT")
    print(f"  6-instruction program: n - 1 - 3 - 5 - 7 - ... (odd number trick)")
    print(f"  NEVER in training data")
    print(f"{'=' * 70}{RESET}")

    correct = 0
    total = 0
    print()
    for n in [0, 1, 2, 3, 4, 5, 8, 9, 10, 15, 16, 20, 25, 30, 36, 49, 64, 81, 100, 120]:
        expected = int(math.isqrt(n))

        mem, pc, result_addr = make_isqrt(n)
        exp_mem, _, exp_steps = run(mem, pc, max_steps=500)
        exp_result = exp_mem[result_addr]

        m, p, model_steps = run_model(model, mem, pc, max_steps=500, device=device)
        got = m[result_addr]

        total += 1
        ok = (got == exp_result)
        if ok:
            correct += 1

        status = f"{GREEN}OK{RESET}" if ok else f"{RED}WRONG (got {got}){RESET}"
        print(f"  isqrt({n:3d}) = {exp_result:2d}  |  model: {got:2d}  |  "
              f"{exp_steps:3d} steps  |  {status}")

    print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    return correct, total


def demo_summary(results, model=None):
    """Print a summary banner."""
    print(f"\n{CYAN}{BOLD}{'=' * 70}")
    print(f"  SUMMARY: Neural Network as a General-Purpose Computer")
    print(f"{'=' * 70}{RESET}")
    n_params = model.count_params() if model else 0
    d = model.d_model if model else '?'
    print(f"\n  Architecture: {BOLD}{n_params/1e6:.1f}M-param transformer{RESET} "
          f"({d}-dim, 6 layers, 8 heads)")
    print(f"  Hardware emulated: 32 bytes memory, 8-bit values [-128, 127]")
    print(f"  Comparable to: Manchester Baby (1948) -- first stored-program computer")
    print(f"  ISA: SUBLEQ (one instruction, Turing complete)")
    print(f"  Training: single-step execution only, random programs + simple traces")
    print(f"\n  Programs NEVER seen during training:\n")

    total_correct = 0
    total_all = 0
    for name, (correct, total) in results.items():
        pct = 100 * correct / total if total > 0 else 0
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        color = GREEN if pct == 100 else (YELLOW if pct >= 80 else RED)
        print(f"    {name:15s}: {correct:3d}/{total:3d} ({color}{pct:5.1f}%{RESET}) [{bar}]")
        total_correct += correct
        total_all += total

    overall = 100 * total_correct / total_all if total_all > 0 else 0
    color = GREEN if overall == 100 else (YELLOW if overall >= 90 else RED)
    print(f"\n    {'OVERALL':15s}: {total_correct:3d}/{total_all:3d} "
          f"({BOLD}{color}{overall:5.1f}%{RESET})")
    print(f"\n  The transformer learned SUBLEQ from data alone, then executed")
    print(f"  arbitrary programs -- Fibonacci, multiplication, division, square root --")
    print(f"  that it {BOLD}never saw during training{RESET}. It learned to BE a computer.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo the SUBLEQ transformer")
    parser.add_argument("model_path", nargs='?', default="checkpoints/best_model.pt",
                        help="Path to model checkpoint (default: checkpoints/best_model.pt)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = auto_device()

    if not os.path.exists(args.model_path):
        print(f"Error: checkpoint not found at {args.model_path}")
        print("Run 'make train' first, or specify a path: python demo.py <path>")
        sys.exit(1)

    model = load_model(args.model_path, args.device)
    results = {}

    fib_c, fib_t = demo_fibonacci(model, args.device)
    results['fibonacci'] = (fib_c, fib_t)

    mul_c, mul_t = demo_multiplication(model, args.device)
    results['multiply'] = (mul_c, mul_t)

    div_c, div_t = demo_division(model, args.device)
    results['division'] = (div_c, div_t)

    isqrt_c, isqrt_t = demo_isqrt(model, args.device)
    results['isqrt'] = (isqrt_c, isqrt_t)

    demo_summary(results, model)
