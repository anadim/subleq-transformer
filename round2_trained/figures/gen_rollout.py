#!/usr/bin/env python3
"""Generate a rollout visualization: 126 ÷ 7 = 18 (91 steps).

Shows memory state evolving step-by-step as the SUBLEQ interpreter
executes an integer division program. This is what "a computer running"
looks like at the memory level.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from subleq import make_div, step, run, MEM_SIZE, CODE_SIZE, DATA_START

# ── Generate the full execution trace ───────────────────────────────
a_val, b_val = 126, 7
mem, pc, result_addr = make_div(a_val, b_val)
expected_result = a_val // b_val

trace = []  # list of (mem_snapshot, pc, step_num)
m, p = list(mem), pc
trace.append((list(m), p, 0))

for s in range(200):
    if p < 0 or p + 2 >= len(m):
        break
    av, bv = m[p], m[p + 1]
    if av < 0 or av >= len(m) or bv < 0 or bv >= len(m):
        break
    new_m, new_p, halted = step(m, p)
    m, p = new_m, new_p
    trace.append((list(m), p, s + 1))
    if halted:
        break

n_steps = len(trace)
print(f"div({a_val}, {b_val}) = {expected_result}, {n_steps - 1} steps")

# ── Figure 1: Memory heatmap over time (data cells only) ───────────
# Show the 5 active data cells evolving over all 91 steps
# Cell 24 = n (dividend, decreasing), 25 = b (divisor, constant)
# Cell 26 = quotient (increasing), 27 = one (constant 1), 29 = temp

active_cells = [24, 25, 26, 27, 29]
cell_labels = ['n (dividend)', 'b (divisor)', 'quotient', 'one', 'temp']

# Also track PC
data = np.zeros((len(active_cells) + 1, n_steps))
for t, (m, p, s) in enumerate(trace):
    data[0, t] = p  # PC
    for i, cell in enumerate(active_cells):
        data[i + 1, t] = m[cell]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 3, 1.5]})

# ── Panel 1: PC over time ──────────────────────────────────────────
ax = axes[0]
steps_x = np.arange(n_steps)
ax.plot(steps_x, data[0], color='#2196F3', linewidth=1.5, alpha=0.8)
ax.fill_between(steps_x, 0, data[0], alpha=0.15, color='#2196F3')
ax.set_ylabel('PC', fontsize=11, fontweight='bold')
ax.set_xlim(0, n_steps - 1)
ax.set_ylim(-2, 16)
ax.set_yticks([0, 3, 6, 9, 12])
ax.tick_params(labelbottom=False)
ax.set_title(f'SUBLEQ Execution: {a_val} ÷ {b_val} = {expected_result}  ({n_steps-1} steps)',
             fontsize=14, fontweight='bold', pad=10)
ax.grid(True, alpha=0.2)

# ── Panel 2: Data cells heatmap ────────────────────────────────────
ax = axes[1]

# Build the heatmap data (cells over time)
heatmap_data = data[1:]  # skip PC row

# Custom colormap: diverging around 0
vmin, vmax = -10, 130
im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
               vmin=vmin, vmax=vmax, interpolation='nearest')
ax.set_yticks(range(len(cell_labels)))
ax.set_yticklabels(cell_labels, fontsize=10)
ax.set_ylabel('Memory cells', fontsize=11, fontweight='bold')
ax.tick_params(labelbottom=False)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
cbar.set_label('Cell value', fontsize=10)

# Annotate key moments
# Every time quotient increments, mark it
for t in range(1, n_steps):
    if heatmap_data[2, t] != heatmap_data[2, t-1]:  # quotient changed
        ax.axvline(x=t, color='white', alpha=0.3, linewidth=0.5)

# ── Panel 3: Quotient & dividend over time ─────────────────────────
ax = axes[2]
ax.plot(steps_x, data[1 + 0], color='#F44336', linewidth=2, label=f'n (dividend)', alpha=0.8)
ax.plot(steps_x, data[1 + 2], color='#4CAF50', linewidth=2, label=f'quotient', alpha=0.8)
ax.axhline(y=expected_result, color='#4CAF50', linestyle='--', alpha=0.4, linewidth=1)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

ax.set_xlabel('Step', fontsize=11, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_xlim(0, n_steps - 1)
ax.legend(loc='center right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2)

# Add annotations
ax.annotate(f'n = {a_val}+1 = {a_val+1}', xy=(0, a_val + 1), fontsize=8,
            color='#F44336', alpha=0.7, ha='left', va='bottom')
ax.annotate(f'quotient = {expected_result}', xy=(n_steps - 1, expected_result),
            fontsize=8, color='#4CAF50', alpha=0.7, ha='right', va='bottom',
            xytext=(-5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'rollout_div126_7.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(os.path.dirname(__file__), 'rollout_div126_7.pdf'),
            bbox_inches='tight', facecolor='white')
print("Saved: figures/rollout_div126_7.png and .pdf")
plt.close()


# ── Figure 2: Memory grid snapshots at key moments ─────────────────
# Show 4x8 grids at steps 0, 20, 45, 91 (start, early, mid, end)
key_steps = [0, 20, 45, n_steps - 1]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

for idx, step_idx in enumerate(key_steps):
    ax = axes[idx]
    m, p, s = trace[step_idx]

    # Build 4x8 grid
    grid = np.array(m).reshape(4, 8)

    # Color: code region dim, data region bright, active instruction highlighted
    colors = np.ones((4, 8, 3)) * 0.95  # light gray background

    for row in range(4):
        for col in range(8):
            cell = row * 8 + col
            if cell < CODE_SIZE:
                # Code region
                if p >= 0 and p <= cell < p + 3:
                    colors[row, col] = [0.6, 0.85, 1.0]  # active instruction: light blue
                else:
                    colors[row, col] = [0.88, 0.88, 0.88]  # code: light gray
            else:
                # Data region
                colors[row, col] = [0.85, 1.0, 0.85]  # data: light green

    ax.imshow(colors, aspect='auto')

    # Write values in cells
    for row in range(4):
        for col in range(8):
            val = grid[row, col]
            cell = row * 8 + col
            fontweight = 'bold' if cell >= DATA_START else 'normal'
            fontsize = 8 if abs(val) < 100 else 7
            color = 'black' if val != 0 else '#999999'
            ax.text(col, row, str(val), ha='center', va='center',
                    fontsize=fontsize, fontweight=fontweight, color=color)

    ax.set_xticks(range(8))
    ax.set_xticklabels(range(8), fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(['0-7', '8-15', '16-23', '24-31'], fontsize=7)
    ax.set_title(f'Step {s} (pc={p})', fontsize=10, fontweight='bold')

    if idx == 0:
        ax.set_ylabel('Memory', fontsize=10)

# Add column labels
fig.suptitle(f'Memory Snapshots: {a_val} ÷ {b_val}  →  quotient grows from 0 to {expected_result}',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'memory_snapshots_div.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(os.path.dirname(__file__), 'memory_snapshots_div.pdf'),
            bbox_inches='tight', facecolor='white')
print("Saved: figures/memory_snapshots_div.png and .pdf")
plt.close()
