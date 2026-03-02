#!/usr/bin/env python3
"""
Run training with detailed logging and generate training curve figures.

Captures per-step:
  - Loss (total, changed positions, unchanged positions)
  - Accuracy (full, changed-only, unchanged-only)
  - Logit confidence (mean log(p/(1-p)) for correct predictions)

Runs TWO configs to compare:
  - Wide: d=256, 6 layers (the winning config, 4.9M params)
  - Deep: d=128, 12 layers (same-ish param count, 2.4M params)

Usage:
    python figures/gen_training_curves.py                  # full run (~45 min)
    python figures/gen_training_curves.py --quick           # quick (~5 min)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import math
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from subleq import MiniSUBLEQTransformer, pregenerate_data
from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE
from subleq.data import CHANGE_WEIGHT


def train_with_logging(config_name, d_model, n_heads, n_layers, d_ff,
                       total_steps, device, seed=42):
    """Train a model and return detailed per-eval-step logs."""
    random.seed(seed)
    torch.manual_seed(seed)

    model = MiniSUBLEQTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, dropout=0.1,
    ).to(device)
    n_params = model.count_params()
    print(f"\n{'='*60}")
    print(f"Training: {config_name} ({n_params:,} params)")
    print(f"  d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"  device={device}, total_steps={total_steps}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    warmup_steps = min(1000, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Curriculum
    def get_instr_range(step):
        frac = step / total_steps
        if frac < 0.10: return (1, 2)
        elif frac < 0.25: return (1, 4)
        elif frac < 0.45: return (1, 6)
        else: return (1, 8)

    batch_size = 256
    data_size = 50000
    log_every = max(1, total_steps // 200)  # ~200 log points
    eval_every = max(1, total_steps // 40)  # ~40 eval points
    regen_every = max(1, total_steps // 20)

    instr_range = get_instr_range(0)
    data_inp, data_out, data_mask = pregenerate_data(data_size, instr_range=instr_range)
    data_inp, data_out, data_mask = data_inp.to(device), data_out.to(device), data_mask.to(device)

    logs = []
    running_loss = 0.0
    running_count = 0
    start_time = time.time()

    for step_num in range(1, total_steps + 1):
        model.train()
        idx = torch.randint(0, data_inp.size(0), (batch_size,))
        inp, out, mask = data_inp[idx], data_out[idx], data_mask[idx]

        logits = model(inp)
        B, S, V = logits.shape
        loss_per_pos = F.cross_entropy(logits.reshape(B*S, V), out.reshape(B*S),
                                       reduction='none').reshape(B, S)
        loss = (loss_per_pos * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_count += 1

        if step_num % regen_every == 0:
            instr_range = get_instr_range(step_num)
            data_inp, data_out, data_mask = pregenerate_data(data_size, instr_range=instr_range)
            data_inp, data_out, data_mask = data_inp.to(device), data_out.to(device), data_mask.to(device)

        if step_num % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_inp, eval_out, eval_mask = pregenerate_data(2000, instr_range=(1, 8))
                eval_inp, eval_out, eval_mask = eval_inp.to(device), eval_out.to(device), eval_mask.to(device)

                eval_logits = model(eval_inp)
                eB, eS, eV = eval_logits.shape

                # Per-position loss
                eval_loss_pp = F.cross_entropy(eval_logits.reshape(eB*eS, eV),
                                               eval_out.reshape(eB*eS),
                                               reduction='none').reshape(eB, eS)

                # Separate changed vs unchanged
                changed_mask = eval_mask > 1.0
                unchanged_mask = ~changed_mask

                total_loss = (eval_loss_pp * eval_mask).sum() / eval_mask.sum()
                changed_loss = (eval_loss_pp[changed_mask]).mean() if changed_mask.any() else torch.tensor(0.0)
                unchanged_loss = (eval_loss_pp[unchanged_mask]).mean() if unchanged_mask.any() else torch.tensor(0.0)

                # Accuracy
                preds = eval_logits.argmax(dim=-1)
                correct = (preds == eval_out)
                full_acc = correct.all(dim=1).float().mean().item()
                changed_acc = (correct | unchanged_mask).all(dim=1).float().mean().item()

                # Per-position accuracy for changed vs unchanged
                changed_pos_acc = correct[changed_mask].float().mean().item() if changed_mask.any() else 1.0
                unchanged_pos_acc = correct[unchanged_mask].float().mean().item() if unchanged_mask.any() else 1.0

                # Logit confidence: log(p/(1-p)) for correct class
                probs = F.softmax(eval_logits, dim=-1)
                correct_probs = probs.gather(-1, eval_out.unsqueeze(-1)).squeeze(-1)
                # Clamp to avoid log(0) or log(inf)
                correct_probs = correct_probs.clamp(1e-7, 1 - 1e-7)
                logit_confidence = torch.log(correct_probs / (1 - correct_probs))

                mean_logit_conf = logit_confidence.mean().item()
                changed_logit_conf = logit_confidence[changed_mask].mean().item() if changed_mask.any() else 0.0
                unchanged_logit_conf = logit_confidence[unchanged_mask].mean().item() if unchanged_mask.any() else 0.0

            entry = {
                'step': step_num,
                'train_loss': running_loss / running_count,
                'eval_loss': total_loss.item(),
                'changed_loss': changed_loss.item(),
                'unchanged_loss': unchanged_loss.item(),
                'full_acc': full_acc,
                'changed_acc': changed_acc,
                'changed_pos_acc': changed_pos_acc,
                'unchanged_pos_acc': unchanged_pos_acc,
                'mean_logit_conf': mean_logit_conf,
                'changed_logit_conf': changed_logit_conf,
                'unchanged_logit_conf': unchanged_logit_conf,
                'lr': optimizer.param_groups[0]['lr'],
                'elapsed': time.time() - start_time,
            }
            logs.append(entry)
            running_loss = 0.0
            running_count = 0

            print(f"  [{config_name}] step {step_num:5d}/{total_steps} | "
                  f"loss={entry['eval_loss']:.3f} | "
                  f"acc={entry['full_acc']:.3f} | "
                  f"changed={entry['changed_pos_acc']:.3f} | "
                  f"logit_conf={entry['mean_logit_conf']:.1f} | "
                  f"{entry['elapsed']:.0f}s")

    return logs, n_params


def plot_figures(wide_logs, deep_logs, wide_params, deep_params, outdir):
    """Generate all four figures."""
    os.makedirs(outdir, exist_ok=True)

    w = {k: [e[k] for e in wide_logs] for k in wide_logs[0]}
    d = {k: [e[k] for e in deep_logs] for k in deep_logs[0]}

    # ── Figure 2: Training loss curves (wide vs deep) ──────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(w['step'], w['eval_loss'], color='#2196F3', linewidth=2,
            label=f'Wide (d=256, 6L, {wide_params/1e6:.1f}M)', alpha=0.9)
    ax.plot(d['step'], d['eval_loss'], color='#F44336', linewidth=2,
            label=f'Deep (d=128, 12L, {deep_params/1e6:.1f}M)', alpha=0.9)
    ax.set_xlabel('Training step', fontsize=12)
    ax.set_ylabel('Eval loss (weighted)', fontsize=12)
    ax.set_title('Training Loss: Wide vs Deep Architecture', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'loss_wide_vs_deep.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'loss_wide_vs_deep.pdf'), bbox_inches='tight')
    print("Saved: loss_wide_vs_deep.png")
    plt.close()

    # ── Figure 3: Logit confidence (saturation diagnostic) ─────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(w['step'], w['changed_logit_conf'], color='#F44336', linewidth=2,
            label='Changed positions', alpha=0.9)
    ax.plot(w['step'], w['unchanged_logit_conf'], color='#4CAF50', linewidth=2,
            label='Unchanged positions', alpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Training step', fontsize=11)
    ax.set_ylabel('log(p / (1-p))  [logit confidence]', fontsize=11)
    ax.set_title(f'Wide model (d=256, 6L)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(d['step'], d['changed_logit_conf'], color='#F44336', linewidth=2,
            label='Changed positions', alpha=0.9)
    ax.plot(d['step'], d['unchanged_logit_conf'], color='#4CAF50', linewidth=2,
            label='Unchanged positions', alpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Training step', fontsize=11)
    ax.set_ylabel('log(p / (1-p))', fontsize=11)
    ax.set_title(f'Deep model (d=128, 12L)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Logit-Space Confidence: How Saturated Are the Predictions?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'logit_saturation.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'logit_saturation.pdf'), bbox_inches='tight')
    print("Saved: logit_saturation.png")
    plt.close()

    # ── Figure 4: Changed vs unchanged accuracy ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(w['step'], [100*x for x in w['changed_pos_acc']], color='#F44336',
            linewidth=2, label='Changed positions', alpha=0.9)
    ax.plot(w['step'], [100*x for x in w['unchanged_pos_acc']], color='#4CAF50',
            linewidth=2, label='Unchanged positions', alpha=0.9)
    ax.plot(w['step'], [100*x for x in w['full_acc']], color='#2196F3',
            linewidth=2, label='Full sequence', alpha=0.9, linestyle='--')
    ax.set_xlabel('Training step', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Wide (d=256, 6L, {wide_params/1e6:.1f}M)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 105)

    ax = axes[1]
    ax.plot(d['step'], [100*x for x in d['changed_pos_acc']], color='#F44336',
            linewidth=2, label='Changed positions', alpha=0.9)
    ax.plot(d['step'], [100*x for x in d['unchanged_pos_acc']], color='#4CAF50',
            linewidth=2, label='Unchanged positions', alpha=0.9)
    ax.plot(d['step'], [100*x for x in d['full_acc']], color='#2196F3',
            linewidth=2, label='Full sequence', alpha=0.9, linestyle='--')
    ax.set_xlabel('Training step', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Deep (d=128, 12L, {deep_params/1e6:.1f}M)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 105)

    fig.suptitle('Per-Position Accuracy: Changed vs Unchanged',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'accuracy_changed_vs_unchanged.png'),
                dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'accuracy_changed_vs_unchanged.pdf'),
                bbox_inches='tight')
    print("Saved: accuracy_changed_vs_unchanged.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 5K steps instead of 20K")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    total_steps = 5000 if args.quick else 20000
    outdir = os.path.dirname(__file__)

    # Train wide model
    wide_logs, wide_params = train_with_logging(
        "wide", d_model=256, n_heads=8, n_layers=6, d_ff=1024,
        total_steps=total_steps, device=args.device, seed=42)

    # Train deep model
    deep_logs, deep_params = train_with_logging(
        "deep", d_model=128, n_heads=8, n_layers=12, d_ff=512,
        total_steps=total_steps, device=args.device, seed=42)

    # Save raw logs
    log_path = os.path.join(outdir, 'training_logs.json')
    with open(log_path, 'w') as f:
        json.dump({'wide': wide_logs, 'deep': deep_logs,
                   'wide_params': wide_params, 'deep_params': deep_params}, f, indent=2)
    print(f"\nSaved raw logs to {log_path}")

    # Generate figures
    plot_figures(wide_logs, deep_logs, wide_params, deep_params, outdir)
    print("\nAll figures generated!")
