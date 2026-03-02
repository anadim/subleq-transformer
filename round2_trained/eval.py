#!/usr/bin/env python3
"""
Evaluate a trained SUBLEQ transformer.

Tests:
1. Single-step accuracy on random states (1-8 instructions)
2. Multi-step programs: negate, addition, countdown, multiply, Fibonacci
3. Error analysis

Usage:
    python eval.py                              # default checkpoint
    python eval.py checkpoints/best_model.pt    # specific checkpoint
    python eval.py --quick                      # fewer examples

Exit code 0 if accuracy >= 99%, 1 otherwise.
"""

import sys
import os
import argparse
import random

import torch

from subleq import (
    MiniSUBLEQTransformer, step, run,
    MEM_SIZE, VALUE_MIN, VALUE_MAX,
    encode, decode, SEQ_LEN, VOCAB_SIZE,
    make_negate, make_addition, make_countdown, make_multiply,
    make_fibonacci, make_halt,
    generate_random_state, generate_random_program,
    pregenerate_data,
)


def auto_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(path, device='cpu'):
    """Load a trained byte-level model from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = MiniSUBLEQTransformer(
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    step_num = ckpt.get('step', '?')
    acc = ckpt.get('best_acc', '?')
    print(f"Loaded model from step {step_num}, best_acc={acc}")
    print(f"Config: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}, "
          f"n_heads={config.get('n_heads')}, d_ff={config.get('d_ff')}")
    print(f"Parameters: {model.count_params():,}")
    return model


def model_step(model, memory, pc, device='cpu'):
    """Use the model to predict one SUBLEQ step."""
    inp = encode(memory, pc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
    pred_tokens = logits.argmax(dim=-1).squeeze(0)
    new_mem, new_pc = decode(pred_tokens)
    return new_mem, new_pc


def eval_single_step(model, n_examples=4000, device='cpu'):
    """Evaluate single-step accuracy across instruction counts."""
    print(f"\n{'='*60}")
    print(f"Single-step evaluation ({n_examples} examples)")
    print(f"{'='*60}")

    all_correct = 0
    all_total = 0

    for n_instr in [1, 2, 3, 4, 5, 6, 7, 8]:
        correct = 0
        changed_correct = 0
        total = 0

        for _ in range(n_examples // 8):
            mem, pc = generate_random_state(n_instr)
            new_mem, new_pc, halted = step(mem, pc)
            if halted:
                continue

            pred_mem, pred_pc = model_step(model, mem, pc, device)
            total += 1

            if pred_mem == new_mem and pred_pc == new_pc:
                correct += 1

            b = mem[pc + 1]
            if 0 <= b < MEM_SIZE and pred_pc == new_pc and pred_mem[b] == new_mem[b]:
                changed_correct += 1

        all_correct += correct
        all_total += total

        if total > 0:
            print(f"  {n_instr}-instr: {correct}/{total} full ({100*correct/total:.1f}%), "
                  f"{changed_correct}/{total} changed ({100*changed_correct/total:.1f}%)")

    if all_total > 0:
        print(f"\n  Overall: {all_correct}/{all_total} ({100*all_correct/all_total:.1f}%)")

    # Batched evaluation
    inp, out, mask = pregenerate_data(n_examples, instr_range=(1, 8))
    inp = inp.to(device)
    out = out.to(device)

    with torch.no_grad():
        logits = model(inp)
    preds = logits.argmax(dim=-1)

    full_correct = (preds == out).all(dim=1).float().mean().item()
    per_pos = (preds == out).float().mean(dim=0)

    print(f"\n  Batched full-step accuracy: {100*full_correct:.1f}%")
    print(f"  Per-position accuracy (byte tokens):")
    print(f"    PC byte:    {100*per_pos[0].item():.1f}%")
    print(f"    Mem avg:    {100*per_pos[1:].mean().item():.1f}%")

    worst = []
    for i in range(SEQ_LEN):
        acc = per_pos[i].item()
        if acc < 0.99:
            label = "PC" if i == 0 else f"mem[{i-1}]"
            worst.append((label, i, acc))
    if worst:
        print(f"\n  Positions below 99%:")
        for label, i, acc in sorted(worst, key=lambda x: x[2]):
            print(f"    {label} (pos {i}): {100*acc:.1f}%")

    return full_correct


def eval_multi_step(model, device='cpu'):
    """Evaluate multi-step program execution."""
    print(f"\n{'='*60}")
    print("Multi-step program evaluation")
    print(f"{'='*60}")

    results = {}

    # --- NEGATE ---
    print("\n  NEGATE (val -> -val, 3 steps):")
    negate_correct = 0
    negate_total = 0
    negate_fails = []
    for val in range(-100, 101):
        mem, pc, result_addr = make_negate(val)
        expected_mem, expected_pc, _ = run(mem, pc, max_steps=10)
        expected_result = expected_mem[result_addr]

        m, p = list(mem), pc
        for s in range(10):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p, device)

        negate_total += 1
        if m[result_addr] == expected_result:
            negate_correct += 1
        else:
            negate_fails.append((val, expected_result, m[result_addr]))

    print(f"    Result: {negate_correct}/{negate_total} "
          f"({100*negate_correct/negate_total:.1f}%)")
    if negate_fails and len(negate_fails) <= 10:
        for val, exp, got in negate_fails:
            print(f"    FAIL: negate({val}) = expected {exp}, got {got}")
    results['negate'] = (negate_correct, negate_total)

    # --- ADDITION ---
    print("\n  ADDITION (a + b, 4 steps):")
    add_correct = 0
    add_total = 0
    add_fails = []
    test_pairs = [(a, b) for a in range(-50, 51, 5) for b in range(-50, 51, 5)]
    random.shuffle(test_pairs)
    for a, b in test_pairs[:300]:
        mem, pc, result_addr = make_addition(a, b)
        expected_mem, expected_pc, _ = run(mem, pc, max_steps=10)
        expected_result = expected_mem[result_addr]

        m, p = list(mem), pc
        for s in range(10):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p, device)

        add_total += 1
        if m[result_addr] == expected_result:
            add_correct += 1
        else:
            add_fails.append((a, b, expected_result, m[result_addr]))

    print(f"    Result: {add_correct}/{add_total} "
          f"({100*add_correct/add_total:.1f}%)")
    if add_fails and len(add_fails) <= 10:
        for a, b, exp, got in add_fails:
            print(f"    FAIL: {a} + {b} = expected {exp}, got {got}")
    results['addition'] = (add_correct, add_total)

    # --- COUNTDOWN ---
    print("\n  COUNTDOWN (n -> 0, ~2n steps):")
    count_correct = 0
    count_total = 0
    for n in range(1, 21):
        mem, pc, result_addr = make_countdown(n)
        expected_mem, expected_pc, expected_steps = run(mem, pc, max_steps=100)

        m, p = list(mem), pc
        for s in range(100):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p, device)

        count_total += 1
        if m[result_addr] == expected_mem[result_addr]:
            count_correct += 1
            print(f"    countdown({n:2d}): OK ({expected_steps} steps)")
        else:
            print(f"    countdown({n:2d}): FAIL - expected {expected_mem[result_addr]}, "
                  f"got {m[result_addr]} (after {s+1} model steps)")

    print(f"    Result: {count_correct}/{count_total} "
          f"({100*count_correct/count_total:.1f}%)")
    results['countdown'] = (count_correct, count_total)

    # --- MULTIPLY ---
    print("\n  MULTIPLY (a * b, 3*b steps) -- NEVER IN TRAINING DATA:")
    mul_correct = 0
    mul_total = 0
    mul_fails = []
    for a in range(1, 13):
        for b in range(1, min(13, VALUE_MAX // max(a, 1) + 1)):
            if a * b > VALUE_MAX:
                continue
            mem, pc, result_addr = make_multiply(a, b)
            expected_mem, expected_pc, expected_steps = run(mem, pc, max_steps=200)
            expected_result = expected_mem[result_addr]

            m, p = list(mem), pc
            for s in range(200):
                if p < 0 or p + 2 >= len(m):
                    break
                m, p = model_step(model, m, p, device)

            mul_total += 1
            if m[result_addr] == expected_result:
                mul_correct += 1
            else:
                mul_fails.append((a, b, expected_result, m[result_addr], expected_steps))

    print(f"    Result: {mul_correct}/{mul_total} "
          f"({100*mul_correct/mul_total:.1f}%)")
    if mul_fails:
        print(f"    Failures ({len(mul_fails)}):")
        for a, b, exp, got, steps in mul_fails[:15]:
            print(f"      {a} * {b} = expected {exp}, got {got} ({steps} steps)")
    results['multiply'] = (mul_correct, mul_total)

    # --- FIBONACCI ---
    print("\n  FIBONACCI (F(2n), F(2n+1)) -- NEVER IN TRAINING DATA:")
    fib_correct = 0
    fib_total = 0
    fib_results = []

    for n in range(1, 7):
        mem, pc, addr_a, addr_b = make_fibonacci(n)
        expected_mem, expected_pc, expected_steps = run(mem, pc, max_steps=500)
        exp_a = expected_mem[addr_a]
        exp_b = expected_mem[addr_b]

        m, p = list(mem), pc
        for s in range(500):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p, device)

        got_a = m[addr_a]
        got_b = m[addr_b]
        fib_total += 1
        ok_a = (got_a == exp_a)
        ok_b = (got_b == exp_b)
        if ok_a and ok_b:
            fib_correct += 1

        fib_results.append((n, exp_a, exp_b, got_a, got_b, expected_steps, ok_a and ok_b))

    for n, exp_a, exp_b, got_a, got_b, steps, ok in fib_results:
        status = "OK" if ok else "FAIL"
        print(f"    n={n}: F({2*n})={exp_a}, F({2*n+1})={exp_b} | "
              f"model: F({2*n})={got_a}, F({2*n+1})={got_b} | "
              f"{steps} steps | {status}")

    print(f"    Result: {fib_correct}/{fib_total} "
          f"({100*fib_correct/fib_total:.1f}%)")
    results['fibonacci'] = (fib_correct, fib_total)

    # --- RANDOM PROGRAMS ---
    print("\n  RANDOM PROGRAMS (up to 30 steps):")
    random.seed(42)
    prog_correct = 0
    prog_total = 0
    for _ in range(100):
        n_instr = random.randint(1, 5)
        mem, pc = generate_random_program(n_instr)
        expected_mem, expected_pc, expected_steps = run(mem, pc, max_steps=30)

        m, p = list(mem), pc
        for s in range(30):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p, device)

        prog_total += 1
        if m == expected_mem and p == expected_pc:
            prog_correct += 1

    print(f"    Result: {prog_correct}/{prog_total} "
          f"({100*prog_correct/prog_total:.1f}%)")
    results['random'] = (prog_correct, prog_total)

    return results


def eval_error_analysis(model, n_examples=2000, device='cpu'):
    """Analyze what kinds of errors the model makes."""
    print(f"\n{'='*60}")
    print("Error analysis")
    print(f"{'='*60}")

    errors = {'pc_wrong': 0, 'mem_wrong': 0, 'both_wrong': 0, 'correct': 0}

    for _ in range(n_examples):
        mem, pc = generate_random_state(random.randint(1, 8))
        new_mem, new_pc, halted = step(mem, pc)
        if halted:
            continue

        pred_mem, pred_pc = model_step(model, mem, pc, device)

        pc_ok = pred_pc == new_pc
        mem_ok = pred_mem == new_mem

        if pc_ok and mem_ok:
            errors['correct'] += 1
        elif pc_ok and not mem_ok:
            errors['mem_wrong'] += 1
        elif not pc_ok and mem_ok:
            errors['pc_wrong'] += 1
        else:
            errors['both_wrong'] += 1

    total = sum(errors.values())
    for k, v in errors.items():
        print(f"  {k}: {v}/{total} ({100*v/total:.1f}%)")

    # Per-cell error breakdown
    print(f"\n  Per-cell error rate (cells that actually change):")
    cell_errors = {}
    cell_total = {}
    for _ in range(n_examples):
        mem, pc = generate_random_state(random.randint(1, 8))
        new_mem, new_pc, halted = step(mem, pc)
        if halted:
            continue

        pred_mem, pred_pc = model_step(model, mem, pc, device)
        b = mem[pc + 1]
        if 0 <= b < MEM_SIZE:
            cell_total[b] = cell_total.get(b, 0) + 1
            if pred_mem[b] != new_mem[b]:
                cell_errors[b] = cell_errors.get(b, 0) + 1

    for cell in sorted(cell_total.keys()):
        errs = cell_errors.get(cell, 0)
        tot = cell_total[cell]
        if errs > 0:
            region = "code" if cell < 24 else "data"
            print(f"    cell {cell:2d} ({region}): {errs}/{tot} errors ({100*errs/tot:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a SUBLEQ transformer")
    parser.add_argument("model_path", nargs='?', default="checkpoints/best_model.pt",
                        help="Path to model checkpoint (default: checkpoints/best_model.pt)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true", help="Quick eval (fewer examples)")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = auto_device()

    if not os.path.exists(args.model_path):
        print(f"Error: checkpoint not found at {args.model_path}")
        print("Run 'make train' first, or specify a path: python eval.py <path>")
        sys.exit(1)

    model = load_model(args.model_path, args.device)

    n = 1000 if args.quick else 4000
    single_step_acc = eval_single_step(model, n_examples=n, device=args.device)
    multi_results = eval_multi_step(model, device=args.device)
    eval_error_analysis(model, n_examples=n, device=args.device)

    # Exit code: 0 if >= 99% single-step accuracy
    sys.exit(0 if single_step_acc >= 0.99 else 1)
