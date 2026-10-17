"""
Figure 2: share of negative, zero and positive scores at each star rating, for the overall
polarity M and the service polarity X (lexicon scorer, proposed dictionary).

Design: two panels of 100% stacked horizontal bars (negative left, zero middle, positive right).
Colours are a diverging pair (red / blue) with a grey midpoint; negative segments are also
hatched so the figure stays readable in black-and-white print. Percentages are printed inside
segments of at least 8%.

Outputs: ../../manuscript/figures/ (or data/output/figures/) figure2_sentiment_by_rating.{png,pdf,svg}
Run from the pipeline root:  python experiments/figure2.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The manuscript folder when the pipeline sits in the authors' project; otherwise (for example in
# the supplementary archive) data/output/figures.
_MANUSCRIPT_FIGS = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'manuscript', 'figures'))
OUT_DIR = _MANUSCRIPT_FIGS if os.path.isdir(_MANUSCRIPT_FIGS) else os.path.join(BASE_DIR, 'data', 'output', 'figures')

NEG, ZERO, POS = '#e34948', '#bab9b3', '#2a78d6'
INK, MUTED = '#0b0b0b', '#52514e'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'hatch.color': 'white', 'hatch.linewidth': 0.8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
})


def shares(df, col):
    ct = pd.crosstab(df['review_rating'], df[col], normalize='index') * 100
    ct = ct.reindex(columns=[-1, 0, 1], fill_value=0)
    return ct.sort_index(ascending=True)


def panel(ax, ct, counts, title, zero_label):
    ratings = list(ct.index)
    y = range(len(ratings))
    left = [0.0] * len(ratings)
    for cls, colour, hatch in ((-1, NEG, '////'), (0, ZERO, None), (1, POS, None)):
        vals = ct[cls].tolist()
        ax.barh(y, vals, left=left, height=0.62, color=colour, hatch=hatch,
                edgecolor='white', linewidth=1.0)
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(left[i] + v / 2, i, f'{v:.0f}%', ha='center', va='center', fontsize=7,
                        color='white' if cls != 0 else INK,
                        bbox=dict(boxstyle='round,pad=0.3', fc=colour, ec='none') if cls == -1 else None)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{r} star{'s' if r > 1 else ''} (n = {counts[r]})" for r in ratings])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color=MUTED)
    ax.tick_params(axis='both', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=9, color=INK, loc='left', pad=6)
    ax.set_xlabel(f'Share of reviews (zero = {zero_label})', color=MUTED)


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'output', 'reviews_scored.csv'))
    counts = df['review_rating'].value_counts().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    panel(axes[0], shares(df, 'customer_satisfaction_M'), counts, '(a) Overall polarity M', 'balanced')
    panel(axes[1], shares(df, 'service_quality_X'), counts, '(b) Service polarity X',
          'no service term or balanced')
    axes[1].tick_params(axis='y', labelleft=False)
    fig.legend(handles=[Patch(facecolor=NEG, hatch='////', edgecolor='white', label='Negative'),
                        Patch(facecolor=ZERO, label='Zero'),
                        Patch(facecolor=POS, label='Positive')],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.5)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in (('png', {'dpi': 600}), ('pdf', {}), ('svg', {})):
        fig.savefig(os.path.join(OUT_DIR, f'figure2_sentiment_by_rating.{ext}'), bbox_inches='tight', **kw)
    print('saved to', OUT_DIR)
    print(shares(df, 'customer_satisfaction_M').round(1).to_string())
    print(shares(df, 'service_quality_X').round(1).to_string())


if __name__ == '__main__':
    main()
