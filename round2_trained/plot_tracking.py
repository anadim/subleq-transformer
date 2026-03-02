#!/usr/bin/env python3
"""Plot per-tier accuracy vs training step from eval_tracking.csv."""

import csv
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)

LOG_PATH = "eval_tracking.csv"

steps = []
tiers = {}

with open(LOG_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row['step']))
        for key in ['single_step', 'negate', 'addition', 'multiply',
                     'fibonacci', 'division', 'sqrt', 'random_multi']:
            if key not in tiers:
                tiers[key] = []
            tiers[key].append(float(row[key]))

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

colors = {
    'single_step': '#2196F3',
    'negate': '#4CAF50',
    'addition': '#8BC34A',
    'multiply': '#FF9800',
    'fibonacci': '#F44336',
    'division': '#9C27B0',
    'sqrt': '#E91E63',
    'random_multi': '#607D8B',
}

labels = {
    'single_step': 'Single-step (500)',
    'negate': 'Negate (21)',
    'addition': 'Addition (121)',
    'multiply': 'Multiply† (up to 200 steps)',
    'fibonacci': 'Fibonacci† (up to 39 steps)',
    'division': 'Division† (up to 91 steps)',
    'sqrt': 'Square root† (up to 61 steps)',
    'random_multi': 'Random multi-step (50)',
}

for key in ['single_step', 'negate', 'addition', 'multiply',
            'fibonacci', 'division', 'sqrt', 'random_multi']:
    ax.plot(steps, tiers[key], 'o-', color=colors[key], label=labels[key],
            markersize=5, linewidth=2)

ax.set_xlabel('Training Step', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Per-Tier Accuracy During Training', fontsize=15)
ax.set_ylim(-5, 105)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# Mark curriculum transitions
for s, label in [(8000, '1-2→1-4'), (20000, '1-4→1-6'), (36000, '1-6→1-8')]:
    if s <= max(steps):
        ax.axvline(x=s, color='gray', linestyle='--', alpha=0.5)
        ax.text(s, 102, label, ha='center', fontsize=8, color='gray')

ax.text(0.02, 0.02, '† = never in training data', transform=ax.transAxes,
        fontsize=9, color='gray', style='italic')

plt.tight_layout()
plt.savefig('eval_tracking.png', dpi=150)
print(f"Saved eval_tracking.png ({len(steps)} data points)")
plt.show()
