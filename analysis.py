import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Sample size and z-score for 95% CI
n = 18379
z = 1.96

# Scores for each metric, model, and regime
data = {
    'BLEU-1': {
        'T5 Original': 0.4531, 'BART Original': 0.4163,
        'T5 Transfer': 0.4581, 'BART Transfer': 0.4265
    },
    'BLEU-2': {
        'T5 Original': 0.2781, 'BART Original': 0.2422,
        'T5 Transfer': 0.2803, 'BART Transfer': 0.2524
    },
    'BLEU-3': {
        'T5 Original': 0.1723, 'BART Original': 0.1379,
        'T5 Transfer': 0.1703, 'BART Transfer': 0.1482
    },
    'BLEU-4': {
        'T5 Original': 0.1074, 'BART Original': 0.0791,
        'T5 Transfer': 0.1027, 'BART Transfer': 0.0882
    },
    'ROUGE-1 P': {
        'T5 Original': 0.3219, 'BART Original': 0.3328,
        'T5 Transfer': 0.3281, 'BART Transfer': 0.3412
    },
    'ROUGE-1 R': {
        'T5 Original': 0.2844, 'BART Original': 0.3032,
        'T5 Transfer': 0.2935, 'BART Transfer': 0.3146
    },
    'ROUGE-1 F': {
        'T5 Original': 0.3002, 'BART Original': 0.3163,
        'T5 Transfer': 0.3080, 'BART Transfer': 0.3269
    },
    'WER': {
        'T5 Original': 0.7726, 'BART Original': 0.7515,
        'T5 Transfer': 0.7714, 'BART Transfer': 0.7471
    }
}

# Define labels and colors
labels = ['T5 Original', 'T5 Transfer', 'BART Original', 'BART Transfer']
color_map = {
    'Original': 'lightcoral',
    'Transfer': 'lightblue'
}
bar_colors = [color_map['Original'] if 'Original' in lbl else color_map['Transfer'] for lbl in labels]
legend_handles = [
    Patch(facecolor='lightcoral', edgecolor='black', label='Original'),
    Patch(facecolor='lightblue', edgecolor='black', label='Transfer')
]

# Function to plot a set of metrics
def plot_metrics(metrics, cols, title):
    fig, axes = plt.subplots(1, cols, figsize=(5*cols, 5), constrained_layout=True)
    if cols == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        scores = data[metric]
        means = [scores[label] for label in labels]
        se = [np.sqrt(p * (1 - p) / n) for p in means]
        margins = [z * s for s in se]
        ax.bar(np.arange(len(labels)), means, color=bar_colors, edgecolor='black', linewidth=1)
        ax.errorbar(np.arange(len(labels)), means, yerr=margins, fmt='none',
                    ecolor='black', elinewidth=3, capsize=8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
    fig.suptitle(title, fontsize=16, y=1.05)
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    plt.show()

# Print CI ranges
print("95% Confidence Intervals:")
for metric, scores in data.items():
    print(f"\n{metric}:")
    for label, p in scores.items():
        se = np.sqrt(p * (1 - p) / n)
        margin = z * se
        lower, upper = p - margin, p + margin
        print(f"  {label}: {lower:.4f} to {upper:.4f}")

# Plot BLEU-N (4)
plot_metrics(['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'], 4, 'BLEU-N Scores Comparison')

# Plot ROUGE-1 (3)
plot_metrics(['ROUGE-1 P', 'ROUGE-1 R', 'ROUGE-1 F'], 3, 'ROUGE-1 Scores Comparison')

# Plot WER alone
plot_metrics(['WER'], 1, 'WER Comparison')
