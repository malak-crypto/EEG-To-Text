import numpy as np
import matplotlib.pyplot as plt

# Sample size and z-score for 95% CI
#n = 18379
n=34678
z = 1.96

# Data definitions
bleu_data = {
    'BLEU-1': {'T5 Orig': 0.4531, 'T5 Trans': 0.4581, 'BART Orig': 0.4163, 'BART Trans': 0.4265},
    'BLEU-2': {'T5 Orig': 0.2781, 'T5 Trans': 0.2803, 'BART Orig': 0.2422, 'BART Trans': 0.2524},
    'BLEU-3': {'T5 Orig': 0.1723, 'T5 Trans': 0.1703, 'BART Orig': 0.1379, 'BART Trans': 0.1482},
    'BLEU-4': {'T5 Orig': 0.1074, 'T5 Trans': 0.1027, 'BART Orig': 0.0791, 'BART Trans': 0.0882}
}
rouge_data = {
    'ROUGE-1 P': {'T5 Orig': 0.3219, 'T5 Trans': 0.3281, 'BART Orig': 0.3328, 'BART Trans': 0.3412},
    'ROUGE-1 R': {'T5 Orig': 0.2844, 'T5 Trans': 0.2935, 'BART Orig': 0.3032, 'BART Trans': 0.3146},
    'ROUGE-1 F': {'T5 Orig': 0.3002, 'T5 Trans': 0.3080, 'BART Orig': 0.3163, 'BART Trans': 0.3269}
}
wer_data = {
    'WER': {'T5 Orig': 0.7726, 'T5 Trans': 0.7714, 'BART Orig': 0.7515, 'BART Trans': 0.7471}
}

def plot_bell(ax, mean, se, label, x_vals):
    density = (1/(se * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_vals-mean)/se)**2)
    lower, upper = mean - z*se, mean + z*se
    ax.plot(x_vals, density, label=label)
    ax.fill_between(x_vals, density, where=(x_vals>=lower)&(x_vals<=upper), alpha=0.2)

# 2x2 BLEU-N
fig_bleu, axes_bleu = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for ax, (metric, scores) in zip(axes_bleu.flat, bleu_data.items()):
    # determine x range
    means = list(scores.values())
    se_vals = [np.sqrt(m*(1-m)/n) for m in means]
    mins = min(m - z*s for m, s in zip(means, se_vals))
    maxs = max(m + z*s for m, s in zip(means, se_vals))
    x = np.linspace(mins-0.005, maxs+0.005, 400)
    for label, mean in scores.items():
        se = np.sqrt(mean*(1-mean)/n)
        plot_bell(ax, mean, se, label, x)
    ax.set_title(metric)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.grid(linestyle='--', linewidth=0.5)
fig_bleu.suptitle('BLEU-N Sampling Distributions with 95% CIs', fontsize=16)
axes_bleu.flat[0].legend(loc='upper right')

# 1x3 ROUGE-1
fig_rouge, axes_rouge = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
for ax, (metric, scores) in zip(axes_rouge, rouge_data.items()):
    means = list(scores.values())
    se_vals = [np.sqrt(m*(1-m)/n) for m in means]
    mins = min(m - z*s for m, s in zip(means, se_vals))
    maxs = max(m + z*s for m, s in zip(means, se_vals))
    x = np.linspace(mins-0.005, maxs+0.005, 400)
    for label, mean in scores.items():
        se = np.sqrt(mean*(1-mean)/n)
        plot_bell(ax, mean, se, label, x)
    ax.set_title(metric)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.grid(linestyle='--', linewidth=0.5)
fig_rouge.suptitle('ROUGE-1 Sampling Distributions with 95% CIs', fontsize=16)
axes_rouge[0].legend(loc='upper right')

# 1x1 WER
fig_wer, ax_wer = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
scores = wer_data['WER']
means = list(scores.values())
se_vals = [np.sqrt(m*(1-m)/n) for m in means]
mins = min(m - z*s for m, s in zip(means, se_vals))
maxs = max(m + z*s for m, s in zip(means, se_vals))
x = np.linspace(mins-0.005, maxs+0.005, 400)
for label, mean in scores.items():
    se = np.sqrt(mean*(1-mean)/n)
    plot_bell(ax_wer, mean, se, label, x)
ax_wer.set_title('WER Sampling Distributions with 95% CIs')
ax_wer.set_xlabel('Score')
ax_wer.set_ylabel('Density')
ax_wer.grid(linestyle='--', linewidth=0.5)
ax_wer.legend(loc='upper right')

plt.show()
