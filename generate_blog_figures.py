#!/usr/bin/env python3
"""Generate 3 blog figures for the 'What Claude Code Built' section."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style setup ──
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'font.family': 'sans-serif',
})

# Color palette
BLUE = '#2563EB'
ORANGE = '#F97316'
GREEN = '#16A34A'
RED = '#DC2626'
PURPLE = '#7C3AED'
GRAY = '#6B7280'
LIGHTBLUE = '#DBEAFE'
LIGHTORANGE = '#FED7AA'
LIGHTGREEN = '#DCFCE7'
LIGHTPURPLE = '#EDE9FE'

# ════════════════════════════════════════════════════════════════
# FIGURE 1: Training curve with curriculum phases
# ════════════════════════════════════════════════════════════════

# Eval data from the paper (wide model, step in thousands)
steps_k = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,
           44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80]
eval_acc = [26.6,47.5,48.9,51.4,76.1,78.9,78.9,80.5,
            69.6,75.4,89.9,89.9,92.2,92.7,94.2,94.5,95.2,
            92.8,99.3,99.1,99.3,99.9,99.9,99.9,100,99.8,99.8,100,
            99.9,99.9,99.9,99.9,100,100,99.8,99.9,100,100,99.9,100]

fig, ax = plt.subplots(figsize=(12, 6))

# Curriculum phase backgrounds
phases = [(0, 8, '1–2 instr', LIGHTBLUE),
          (8, 20, '1–4 instr', LIGHTORANGE),
          (20, 36, '1–6 instr', LIGHTGREEN),
          (36, 82, '1–8 instr', LIGHTPURPLE)]

for start, end, label, color in phases:
    ax.axvspan(start, end, alpha=0.35, color=color, zorder=0)
    mid = (start + end) / 2
    ax.text(mid, 15, label, ha='center', va='center', fontsize=11,
            fontweight='bold', color=GRAY, alpha=0.8)

# Transition lines
for x in [8, 20, 36]:
    ax.axvline(x=x, color=GRAY, linestyle='--', alpha=0.5, linewidth=1)

# Main curve
ax.plot(steps_k, eval_acc, 'o-', color=BLUE, linewidth=2.5, markersize=7,
        markerfacecolor='white', markeredgewidth=2, markeredgecolor=BLUE,
        zorder=5)

# Highlight the 100% points
for s, a in zip(steps_k, eval_acc):
    if a == 100:
        ax.plot(s, a, 'o', color=GREEN, markersize=10, markeredgewidth=2,
                markeredgecolor=GREEN, markerfacecolor=GREEN, zorder=6, alpha=0.7)

# First 100% annotation
ax.annotate('First 100%!', xy=(50, 100), xytext=(42, 88),
            fontsize=13, fontweight='bold', color=GREEN,
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2),
            zorder=7)

# Curriculum dip annotation
ax.annotate('curriculum\ntransition dip', xy=(36, 92.8), xytext=(28, 78),
            fontsize=11, color=ORANGE, fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5),
            zorder=7)

ax.set_xlabel('Training Step (×1000)', fontweight='bold')
ax.set_ylabel('Eval Accuracy (%)', fontweight='bold')
ax.set_title('Learning to Execute SUBLEQ: Training Curve', fontweight='bold', pad=15)
ax.set_xlim(0, 82)
ax.set_ylim(10, 103)
ax.set_yticks([20, 40, 60, 80, 100])

plt.tight_layout()
plt.savefig('/Users/anadim/subleq-transformer/fig1_training_curve.png', dpi=200, bbox_inches='tight')
print("Saved fig1_training_curve.png")
plt.close()

# ════════════════════════════════════════════════════════════════
# FIGURE 2: Width vs Depth scaling
# ════════════════════════════════════════════════════════════════

models = {
    'Tiny\n(d=64, L=2)': {'params': 135, 'acc': 23.3, 'width': 64, 'depth': 2},
    'Small\n(d=64, L=4)': {'params': 235, 'acc': 20.6, 'width': 64, 'depth': 4},
    'Medium\n(d=128, L=4)': {'params': 864, 'acc': 49.1, 'width': 128, 'depth': 4},
    'Base\n(d=128, L=6)': {'params': 1260, 'acc': 36.2, 'width': 128, 'depth': 6},
    'Deep\n(d=128, L=12)': {'params': 2450, 'acc': 74.8, 'width': 128, 'depth': 12},
    'Wide\n(d=256, L=6)': {'params': 4879, 'acc': 95.9, 'width': 256, 'depth': 6},
}

fig, ax = plt.subplots(figsize=(11, 7.5))

names = list(models.keys())
accs = [models[n]['acc'] for n in names]
params = [models[n]['params'] for n in names]
widths = [models[n]['width'] for n in names]

# Color by width
color_map = {64: GRAY, 128: ORANGE, 256: BLUE}
colors = [color_map[w] for w in widths]

bars = ax.bar(range(len(names)), accs, color=colors, edgecolor='white',
              linewidth=2, width=0.7, zorder=3)

# Add param count labels on bars
for i, (bar, p, a) in enumerate(zip(bars, params, accs)):
    label = f"{p/1000:.1f}M" if p >= 1000 else f"{p}K"
    ax.text(bar.get_x() + bar.get_width()/2, a + 1.5, label,
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=colors[i])

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Full Accuracy (%)', fontweight='bold')
ax.set_title('Width Dominates Depth for SUBLEQ Execution', fontweight='bold', pad=15)
ax.set_ylim(0, 108)

# Highlight the comparison
ax.annotate('', xy=(4, 74.8), xytext=(5, 95.9),
            arrowprops=dict(arrowstyle='<->', color=RED, lw=2.5))
ax.text(4.5, 86, '2× params\n+21 pts', ha='center', fontsize=12,
        fontweight='bold', color=RED)

# Subtitle note — placed lower to avoid overlap with x-axis labels
ax.text(0.5, -0.13, 'Same width (d=128), 3× more depth: only +25 pts  |  2× width (d=256): +21 pts over Deep with fewer layers',
        ha='center', fontsize=9.5, color=GRAY, fontstyle='italic',
        transform=ax.transAxes)

# Legend
legend_elements = [mpatches.Patch(facecolor=GRAY, label='d=64'),
                   mpatches.Patch(facecolor=ORANGE, label='d=128'),
                   mpatches.Patch(facecolor=BLUE, label='d=256')]
ax.legend(handles=legend_elements, loc='upper left', title='Model Width',
          title_fontsize=12)

plt.tight_layout()
plt.savefig('/Users/anadim/subleq-transformer/fig2_width_vs_depth.png', dpi=200, bbox_inches='tight')
print("Saved fig2_width_vs_depth.png")
plt.close()

# ════════════════════════════════════════════════════════════════
# FIGURE 3: Multiplication table — the "it works" visual
# ════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 10))

# Build the multiplication table
max_val = 127  # 8-bit signed max
table = np.zeros((12, 12))
mask = np.ones((12, 12), dtype=bool)  # True = valid (not overflow)

for i in range(12):
    for j in range(12):
        val = (i+1) * (j+1)
        if val <= max_val:
            table[i, j] = val
        else:
            mask[i, j] = False

# Create colored grid
for i in range(12):
    for j in range(12):
        a, b = i+1, j+1
        val = a * b
        if val <= max_val:
            # Color intensity by value
            intensity = val / max_val
            color = plt.cm.Blues(0.15 + 0.6 * intensity)
            rect = plt.Rectangle((j, 11-i), 1, 1, facecolor=color,
                                  edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            ax.text(j + 0.5, 11-i + 0.5, str(val),
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if intensity > 0.5 else '#1E3A5F')
        else:
            rect = plt.Rectangle((j, 11-i), 1, 1, facecolor='#F3F4F6',
                                  edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            ax.text(j + 0.5, 11-i + 0.5, '—',
                    ha='center', va='center', fontsize=10, color='#D1D5DB')

# Labels
for i in range(12):
    ax.text(i + 0.5, 12.3, str(i+1), ha='center', va='center',
            fontsize=13, fontweight='bold')
    ax.text(-0.5, 11-i + 0.5, str(i+1), ha='center', va='center',
            fontsize=13, fontweight='bold')

ax.set_xlim(-0.8, 12)
ax.set_ylim(-0.8, 12.8)
ax.set_aspect('equal')
ax.axis('off')

ax.set_title('Multiplication Table Computed by the Trained Transformer\n'
             '141/141 correct — never seen during training',
             fontweight='bold', fontsize=16, pad=20)

# "All correct" badge
badge = mpatches.FancyBboxPatch((3.5, -0.7), 5, 0.6,
                                 boxstyle="round,pad=0.1",
                                 facecolor=GREEN, edgecolor='white',
                                 linewidth=2, alpha=0.9)
ax.add_patch(badge)
ax.text(6, -0.4, '141/141 CORRECT', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')

# Gray = overflow note
ax.text(6, -1.3, 'Gray cells = overflow (exceed 8-bit range), not tested',
        ha='center', fontsize=10, color=GRAY, fontstyle='italic')

plt.tight_layout()
plt.savefig('/Users/anadim/subleq-transformer/fig3_multiplication_table.png', dpi=200, bbox_inches='tight')
print("Saved fig3_multiplication_table.png")
plt.close()

print("\nDone! Three figures saved:")
print("  fig1_training_curve.png       — Training curve with curriculum phases")
print("  fig2_width_vs_depth.png       — Width vs depth scaling comparison")
print("  fig3_multiplication_table.png — 12x12 multiplication table (141/141 correct)")
