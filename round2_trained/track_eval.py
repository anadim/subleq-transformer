#!/usr/bin/env python3
"""
Monitor training progress by periodically evaluating the checkpoint on all tiers.
Runs on CPU to avoid competing with MPS training.
Logs results to eval_tracking.csv for plotting.
"""

import sys
import os
import time
import random
import json
import csv

import torch

from subleq import (
    MiniSUBLEQTransformer, step, run,
    MEM_SIZE, VALUE_MIN, VALUE_MAX,
    encode, decode, SEQ_LEN, VOCAB_SIZE,
    make_negate, make_addition, make_countdown, make_multiply,
    make_fibonacci, make_div, make_isqrt, make_halt,
    generate_random_state, generate_random_program,
    pregenerate_data,
)

CKPT_PATH = "checkpoints/best_model.pt"
LOG_PATH = "eval_tracking.csv"
POLL_INTERVAL = 120  # seconds between checks


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
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
    model.eval()
    return model, ckpt.get('step', 0), ckpt.get('best_acc', 0)


def model_step(model, memory, pc):
    inp = encode(memory, pc).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
    pred_tokens = logits.argmax(dim=-1).squeeze(0)
    new_mem, new_pc = decode(pred_tokens)
    return new_mem, new_pc


def model_run(model, mem, pc, max_steps):
    m, p = list(mem), pc
    for s in range(max_steps):
        if p < 0 or p + 2 >= len(m):
            return m, p, s
        m, p = model_step(model, m, p)
    return m, p, max_steps


def eval_single_step(model, n=500):
    correct = 0
    total = 0
    for n_instr in [1, 2, 3, 4, 5, 6, 7, 8]:
        for _ in range(n // 8):
            mem, pc = generate_random_state(n_instr)
            new_mem, new_pc, halted = step(mem, pc)
            if halted:
                continue
            pred_mem, pred_pc = model_step(model, mem, pc)
            total += 1
            if pred_mem == new_mem and pred_pc == new_pc:
                correct += 1
    return correct, total


def eval_negate(model):
    correct = 0
    total = 0
    for val in range(-100, 101, 10):  # 21 values
        mem, pc, r = make_negate(val)
        gt_mem, gt_pc, _ = run(mem, pc, max_steps=10)
        m, p = list(mem), pc
        for _ in range(10):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p)
        total += 1
        if m[r] == gt_mem[r]:
            correct += 1
    return correct, total


def eval_addition(model):
    correct = 0
    total = 0
    for a in range(-50, 51, 10):
        for b in range(-50, 51, 10):
            if abs(a + b) > VALUE_MAX:
                continue
            mem, pc, r = make_addition(a, b)
            gt_mem, gt_pc, _ = run(mem, pc, max_steps=10)
            m, p = list(mem), pc
            for _ in range(10):
                if p < 0 or p + 2 >= len(m):
                    break
                m, p = model_step(model, m, p)
            total += 1
            if m[r] == gt_mem[r]:
                correct += 1
    return correct, total


def eval_multiply(model):
    correct = 0
    total = 0
    for a in range(1, 12):
        for b in range(1, min(12, VALUE_MAX // max(a, 1) + 1)):
            if a * b > VALUE_MAX:
                continue
            mem, pc, r = make_multiply(a, b)
            gt_mem, _, _ = run(mem, pc, max_steps=200)
            md_mem, _, _ = model_run(model, mem, pc, 200)
            total += 1
            if md_mem[r] == gt_mem[r]:
                correct += 1
    return correct, total


def eval_fibonacci(model):
    correct = 0
    total = 0
    for n in range(1, 6):
        mem, pc, addr_a, addr_b = make_fibonacci(n)
        gt_mem, _, _ = run(mem, pc, max_steps=500)
        md_mem, _, _ = model_run(model, mem, pc, 500)
        total += 1
        if md_mem[addr_a] == gt_mem[addr_a] and md_mem[addr_b] == gt_mem[addr_b]:
            correct += 1
    return correct, total


def eval_division(model):
    correct = 0
    total = 0
    cases = [(10, 2), (15, 5), (100, 10), (99, 9), (126, 7), (100, 13), (50, 8), (77, 7)]
    for a, b in cases:
        if a > VALUE_MAX:
            continue
        mem, pc, r = make_div(a, b)
        gt_mem, _, _ = run(mem, pc, max_steps=200)
        md_mem, _, _ = model_run(model, mem, pc, 200)
        total += 1
        if md_mem[r] == gt_mem[r]:
            correct += 1
    return correct, total


def eval_sqrt(model):
    correct = 0
    total = 0
    for n in [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]:
        if n > VALUE_MAX:
            continue
        mem, pc, r = make_isqrt(n)
        gt_mem, _, _ = run(mem, pc, max_steps=200)
        md_mem, _, _ = model_run(model, mem, pc, 200)
        total += 1
        if md_mem[r] == gt_mem[r]:
            correct += 1
    return correct, total


def eval_random_multistep(model):
    correct = 0
    total = 0
    random.seed(42)
    for _ in range(50):
        n_instr = random.randint(1, 5)
        mem, pc = generate_random_program(n_instr)
        gt_mem, gt_pc, _ = run(mem, pc, max_steps=30)
        md_mem, md_pc, _ = model_run(model, mem, pc, 30)
        total += 1
        if md_mem == gt_mem and md_pc == gt_pc:
            correct += 1
    return correct, total


def run_all_evals(model):
    results = {}
    results['single_step'] = eval_single_step(model)
    results['negate'] = eval_negate(model)
    results['addition'] = eval_addition(model)
    results['multiply'] = eval_multiply(model)
    results['fibonacci'] = eval_fibonacci(model)
    results['division'] = eval_division(model)
    results['sqrt'] = eval_sqrt(model)
    results['random_multi'] = eval_random_multistep(model)
    return results


def main():
    # Write CSV header
    header = ['step', 'best_acc', 'single_step', 'negate', 'addition',
              'multiply', 'fibonacci', 'division', 'sqrt', 'random_multi']
    write_header = not os.path.exists(LOG_PATH)

    if write_header:
        with open(LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    last_mtime = 0
    last_step = -1

    print("Tracking eval progress. Polling every 2 min for checkpoint updates...")
    print(f"Logging to {LOG_PATH}")

    while True:
        if not os.path.exists(CKPT_PATH):
            print("  Waiting for checkpoint...")
            time.sleep(POLL_INTERVAL)
            continue

        mtime = os.path.getmtime(CKPT_PATH)
        if mtime <= last_mtime:
            time.sleep(POLL_INTERVAL)
            continue

        last_mtime = mtime
        print(f"\n  Checkpoint updated at {time.strftime('%H:%M:%S')}, evaluating...")

        try:
            model, step_num, best_acc = load_model(CKPT_PATH)
        except Exception as e:
            print(f"  Error loading: {e}")
            time.sleep(30)
            continue

        if step_num == last_step:
            time.sleep(POLL_INTERVAL)
            continue
        last_step = step_num

        results = run_all_evals(model)

        row = [step_num, f"{best_acc:.4f}"]
        for key in ['single_step', 'negate', 'addition', 'multiply',
                     'fibonacci', 'division', 'sqrt', 'random_multi']:
            c, t = results[key]
            pct = 100 * c / t if t > 0 else 0
            row.append(f"{pct:.1f}")
            print(f"    {key}: {c}/{t} ({pct:.1f}%)")

        with open(LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"  Step {step_num} logged.")
        del model

        # Check if training is done
        try:
            import subprocess
            result = subprocess.run(['pgrep', '-f', 'train.py'], capture_output=True)
            if result.returncode != 0:
                print("\n  Training process finished. Running final eval...")
                model, step_num, best_acc = load_model(CKPT_PATH)
                results = run_all_evals(model)
                row = [step_num, f"{best_acc:.4f}"]
                for key in ['single_step', 'negate', 'addition', 'multiply',
                             'fibonacci', 'division', 'sqrt', 'random_multi']:
                    c, t = results[key]
                    pct = 100 * c / t if t > 0 else 0
                    row.append(f"{pct:.1f}")
                with open(LOG_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print("  Done! Use plot_tracking.py to visualize.")
                break
        except:
            pass

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
