import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Helper to draw boxes
def draw_box(ax, x, y, w, h, text, color='lightblue'):
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Helper to draw arrows
def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# Draw Environment
draw_box(ax, 0.5, 2.5, 2, 1, "RIS Environment\n(N=4, M=64, K=4)", 'lightgreen')

# Draw State
draw_box(ax, 3.5, 2.5, 2, 1, "State Vector\n(1056 dim)", 'lightyellow')

# Draw Models
draw_box(ax, 6.5, 4.0, 2, 1, "M1: DDPG", 'lightblue')
draw_box(ax, 6.5, 2.5, 2, 1, "M2: SAC", 'lightblue')
draw_box(ax, 6.5, 1.0, 2, 1, "M3: Unfolding", 'lightblue')

# Draw Actions / Outputs
draw_box(ax, 9.5, 4.0, 2, 1, "Action (64 dim)\nContinuous", 'lightcoral')
draw_box(ax, 9.5, 2.5, 2, 1, "Action (64 dim)\nDiscrete (3-bit)", 'lightcoral')
draw_box(ax, 9.5, 1.0, 2, 1, "Phase (64 dim)\nAO Target", 'lightcoral')

# Arrows: Env to State
draw_arrow(ax, 2.5, 3.0, 3.5, 3.0)

# Arrows: State to Models
draw_arrow(ax, 5.5, 3.0, 6.5, 4.5)
draw_arrow(ax, 5.5, 3.0, 6.5, 3.0)
draw_arrow(ax, 5.5, 3.0, 6.5, 1.5)

# Arrows: Models to Actions
draw_arrow(ax, 8.5, 4.5, 9.5, 4.5)
draw_arrow(ax, 8.5, 3.0, 9.5, 3.0)
draw_arrow(ax, 8.5, 1.5, 9.5, 1.5)

# Feedback arrow (Action back to Environment)
ax.annotate("Reward (Sum Rate)", xy=(2.5, 2.2), xytext=(5.5, 0.2),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5, connectionstyle="arc3,rad=0.3"))

plt.title("Figure 1. Unified Evaluation Framework Architecture", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure1_architecture.png', dpi=300)
print("Figure 1 saved successfully to figures/figure1_architecture.png")