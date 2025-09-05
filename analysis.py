import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Sample size and z-score for 95% CI
#n = 18379

z = 1.96

# bleu_data = {
#     'BLEU-1': {'T5 Orig': 0.4531, 'T5 Trans': 0.4581, 'BART Orig': 0.4163, 'BART Trans': 0.4265},
#     'BLEU-2': {'T5 Orig': 0.2781, 'T5 Trans': 0.2803, 'BART Orig': 0.2422, 'BART Trans': 0.2524},
#     'BLEU-3': {'T5 Orig': 0.1723, 'T5 Trans': 0.1703, 'BART Orig': 0.1379, 'BART Trans': 0.1482},
#     'BLEU-4': {'T5 Orig': 0.1074, 'T5 Trans': 0.1027, 'BART Orig': 0.0791, 'BART Trans': 0.0882}
# }
# rouge_data = {
#     'ROUGE-1 P': {'T5 Orig': 0.3219, 'T5 Trans': 0.3281, 'BART Orig': 0.3328, 'BART Trans': 0.3412},
#     'ROUGE-1 R': {'T5 Orig': 0.2844, 'T5 Trans': 0.2935, 'BART Orig': 0.3032, 'BART Trans': 0.3146},
#     'ROUGE-1 F': {'T5 Orig': 0.3002, 'T5 Trans': 0.3080, 'BART Orig': 0.3163, 'BART Trans': 0.3269}
# }
# wer_data = {
#     'WER': {'T5 Orig': 0.7726, 'T5 Trans': 0.7714, 'BART Orig': 0.7515, 'BART Trans': 0.7471}
# }

n=34678
bleu_data = {
    'BLEU-1': {'T5 Orig': 0.4422, 'T5 Trans': 0.4462, 'BART Orig': 0.4074, 'BART Trans': 0.4145},
    'BLEU-2': {'T5 Orig': 0.2677, 'T5 Trans': 0.2709, 'BART Orig': 0.2315, 'BART Trans': 0.2398},
    'BLEU-3': {'T5 Orig': 0.1626, 'T5 Trans': 0.1642, 'BART Orig': 0.1299, 'BART Trans': 0.1369},
    'BLEU-4': {'T5 Orig': 0.0994, 'T5 Trans': 0.0992, 'BART Orig': 0.0743, 'BART Trans': 0.0792}
}
rouge_data = {
    'ROUGE-1 P': {'T5 Orig': 0.3162, 'T5 Trans': 0.3224, 'BART Orig': 0.3268, 'BART Trans': 0.3314},
    'ROUGE-1 R': {'T5 Orig': 0.2787, 'T5 Trans': 0.2856, 'BART Orig': 0.2988, 'BART Trans': 0.3073},
    'ROUGE-1 F': {'T5 Orig': 0.2943, 'T5 Trans': 0.3010, 'BART Orig': 0.3112, 'BART Trans': 0.3183}
}
wer_data = {
    'WER': {'T5 Orig': 0.7949, 'T5 Trans': 0.7797, 'BART Orig': 0.7585, 'BART Trans': 0.7539}
}

labels = ['T5 Orig', 'T5 Trans', 'BART Orig', 'BART Trans']
color_map = {'Orig': 'lightcoral', 'Trans': 'lightblue'}
bar_colors = [color_map['Orig'] if 'Orig' in lbl else color_map['Trans'] for lbl in labels]
legend_handles = [
    Patch(facecolor='lightcoral', edgecolor='black', label='Original'),
    Patch(facecolor='lightblue', edgecolor='black', label='Transfer')
]

def plot_metrics(dataset, cols, title):
    fig, axes = plt.subplots(1, cols, figsize=(5*cols, 5), constrained_layout=True)
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, (metric, scores) in zip(axes, dataset.items()):
        means = [scores[l] for l in labels]
        se = [np.sqrt(m*(1-m)/n) for m in means]
        ci = [z*s for s in se]
        ax.bar(range(len(labels)), means, color=bar_colors, edgecolor='black')
        ax.errorbar(range(len(labels)), means, yerr=ci, fmt='none',
                    ecolor='black', elinewidth=2, capsize=6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0,1)
        ax.set_title(metric)
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
    fig.suptitle(title, y=1.05, fontsize=14)
    fig.legend(handles=legend_handles, ncol=2,
               loc='lower center', bbox_to_anchor=(0.5, -0.1))
    plt.show()  # <-- ensure figure renders

# Now call:
plot_metrics(bleu_data, 4, 'BLEU‑N Scores Comparison')
plot_metrics(rouge_data, 3, 'ROUGE‑1 Scores Comparison')
plot_metrics(wer_data, 1, 'WER Comparison')
