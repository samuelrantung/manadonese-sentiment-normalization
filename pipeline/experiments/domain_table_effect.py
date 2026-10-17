"""
Effect of the domain-correction table (Section 2.4) on the lexicon scorer.

Scores the main pipeline's preprocessed text (proposed dictionary) twice, with the same rules:
with the domain table (the pipeline) and with raw InSet net weights. Reports how many reviews
change polarity (compared with the change made by the dictionary), accuracy on the gold standard
(agreed labels, paired cluster bootstrap over restaurants, 2,000 draws, seed 42) and the correlation
with the star rating. The table was compiled while correlations with ratings were being monitored
and frozen before the gold-standard evaluation, so only the gold-set comparison is independent of it.

Report: data/output/domain_table_effect.txt
Run from the pipeline root:  python experiments/domain_table_effect.py
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_DIR = os.path.join(BASE_DIR, 'annotation')
OUTPUT = os.path.join(BASE_DIR, 'data', 'output')
sys.path.insert(0, os.path.join(BASE_DIR, 'steps'))

from step7_sentiment_scoring import (                                    # noqa: E402
    NEG_LEXICON_FILE, POS_LEXICON_FILE, SERVICE_KEYWORDS_FILE, calculate_sentiment_score,
    classify_sentiment, load_lexicon, load_service_keywords, score_absa_service_quality)

REPORT = os.path.join(OUTPUT, 'domain_table_effect.txt')
KEY = ['review_author', 'google_id', 'review_date']
POL = {'Positif': 1, 'Netral-campuran': 0, 'Negatif': -1}
N_BOOT, SEED = 2000, 42


def raw_inset():
    """InSet net weights (positive + negative file), without the domain table."""
    lex = {}
    for path in (POS_LEXICON_FILE, NEG_LEXICON_FILE):
        t = pd.read_csv(path, sep='\t')
        for w, x in zip(t['word'].astype(str).str.strip().str.lower(), t['weight']):
            lex[w] = lex.get(w, 0.0) + float(x)
    return lex


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        tables = {'with domain table': load_lexicon(POS_LEXICON_FILE, NEG_LEXICON_FILE),
                  'raw InSet': raw_inset()}
        keywords = load_service_keywords(SERVICE_KEYWORDS_FILE)
    df = pd.read_csv(os.path.join(OUTPUT, 'reviews_preprocessed.csv'))
    df['review_text'] = df['review_text'].fillna('')
    y = df['review_rating'].astype(float)
    for name, lex in tables.items():
        df[f'm|{name}'] = df['review_text'].map(lambda t: classify_sentiment(calculate_sentiment_score(t, lex)))
        df[f'x|{name}'] = df['review_text'].map(lambda t: score_absa_service_quality(t, keywords, lex))
    abl = pd.read_csv(os.path.join(OUTPUT, 'ablation_predictions.csv'))

    labels = pd.read_csv(os.path.join(ANNOT_DIR, 'merged_labels.csv'))
    key = pd.read_csv(os.path.join(ANNOT_DIR, 'gold_sample_key.csv'))[['review_id'] + KEY]
    g = labels.merge(key, on='review_id').merge(df, on=KEY)
    agree = lambda c: g[f'{c}_A'].where(g[f'{c}_A'] == g[f'{c}_B'])  # noqa: E731
    gold = {'overall': agree('overall_pol').map(POL),
            'service': agree('service_pol').map(POL).where(agree('mention') == 'Ya')}

    out = [f'DOMAIN TABLE EFFECT (n = {len(df):,}; proposed dictionary, r = 5, scope 2)\n']
    p = out.append
    a, b = 'with domain table', 'raw InSet'
    p(f'  Reviews whose M changes because of the domain table: {(df[f"m|{a}"] != df[f"m|{b}"]).sum()}; '
      f'X: {(df[f"x|{a}"] != df[f"x|{b}"]).sum()}')
    p(f'  Reviews whose M changes because of the dictionary (proposed vs none): '
      f'{(abl["m_proposed"] != abl["m_none"]).sum()}; X: {(abl["x_proposed"] != abl["x_none"]).sum()}\n')
    p(f'  {"":20} {"r(M,Y)":>7} {"r(X,Y)":>7}')
    for name in tables:
        p(f'  {name:20} {df[f"m|{name}"].corr(y):7.3f} {df[f"x|{name}"].corr(y):7.3f}')

    rng = np.random.default_rng(SEED)
    p('\n  Gold standard (agreed labels): accuracy, and difference with 95% cluster-bootstrap CI')
    for task, col in (('overall', 'm'), ('service', 'x')):
        t = g[gold[task].notna()].reset_index(drop=True)
        true = gold[task].dropna().to_numpy()
        ids = t['google_id'].unique()
        pos = {i: np.flatnonzero(t['google_id'].values == i) for i in ids}
        pa, pb = t[f'{col}|{a}'].to_numpy(), t[f'{col}|{b}'].to_numpy()
        boot = []
        for _ in range(N_BOOT):
            idx = np.concatenate([pos[i] for i in rng.choice(ids, size=len(ids))])
            boot.append(np.mean(pa[idx] == true[idx]) - np.mean(pb[idx] == true[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p(f'  {task:8} (n = {len(t)}): with table {np.mean(pa == true):.3f}, raw InSet {np.mean(pb == true):.3f}; '
          f'difference {np.mean(pa == true) - np.mean(pb == true):+.3f} [{lo:+.3f}, {hi:+.3f}]')

    text = '\n'.join(out)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
