"""
Normalization ablation and dictionary-robustness check.

Every condition starts from the same archived translation (reviews_translated.csv), applies
one normalization resource, then reuses the pipeline's own preprocessing (Step 5) and
scoring (Step 7) unchanged. Only the normalization resource differs between conditions.

Conditions
  none        : no normalization (the 'salah satu' guard is still applied)
  colloquial  : Colloquial Indonesian Lexicon only (Salsabila et al., 2018)
  proposed    : the 109-entry Manadonese dictionary used in the paper
  combined    : colloquial + proposed (proposed wins on conflicts)
  run1..run3  : the three Gemini 3.8 Flash rerun dictionaries (robustness check)

All conditions are evaluated on the same 1,096 reviews as the main analysis.
Results: data/output/ablation_results.csv

Run from the pipeline root:  python experiments/ablation.py
"""
import csv
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'steps'))

from step5_text_preprocessing import build_stopwords                     # noqa: E402
from step7_sentiment_scoring import (                                    # noqa: E402
    NEGATION_WORDS, NEG_LEXICON_FILE, POS_LEXICON_FILE, SERVICE_KEYWORDS_FILE,
    calculate_sentiment_score, classify_sentiment, load_lexicon,
    load_service_keywords, score_absa_service_quality)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory              # noqa: E402

INPUT   = os.path.join(BASE_DIR, 'data', 'input')
OUTPUT  = os.path.join(BASE_DIR, 'data', 'output')
PROV    = os.path.join(os.path.dirname(BASE_DIR), 'llm_provenance')
KEY     = ['review_author', 'google_id', 'review_date']


# ---------- normalization resources ----------
def load_json(path):
    with open(path, encoding='utf-8') as f:
        return {k.lower(): v.lower() for k, v in json.load(f).items()}


def load_colloquial(path):
    lex = {}
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            slang, formal = r['slang'].strip().lower(), r['formal'].strip().lower()
            if slang and slang != formal:
                lex.setdefault(slang, formal)   # first entry wins on duplicates
    return lex


def resources():
    proposed = load_json(os.path.join(INPUT, 'manado_slang_dict.json'))
    colloquial = load_colloquial(os.path.join(INPUT, 'colloquial-indonesian-lexicon.csv'))
    res = {
        'none': {},
        'colloquial': colloquial,
        'proposed': proposed,
        'combined': {**colloquial, **proposed},
    }
    for i in (1, 2, 3):
        res[f'run{i}'] = load_json(os.path.join(PROV, f'run{i}.json'))
    return res


def normalize(text, lex, pattern):
    """Same logic as Step 4, but also returns the target word of every substitution."""
    text = re.sub(r'\bsalah satu\b', 'satu', str(text).lower())
    if pattern is None:
        return text, []
    targets = []

    def sub(m):
        t = lex[m.group(0).lower()]
        targets.append(t)
        return t
    return pattern.sub(sub, text), targets


# ---------- preprocessing (Step 5 logic, stemming cached per word) ----------
class Preprocessor:
    def __init__(self):
        self.stop = build_stopwords()
        self.stemmer = StemmerFactory().create_stemmer()
        self.cache = {}

    def stem(self, word):
        if word not in self.cache:
            self.cache[word] = self.stemmer.stem(word)
        return self.cache[word]

    def __call__(self, text):
        kept = [w for w in text.split() if w not in self.stop]
        return ' '.join(s for s in (self.stem(w) for w in kept) if s)


# ---------- statistics ----------
def model2(df):
    """Y ~ X + M with standard errors clustered by restaurant."""
    fit = smf.ols('y ~ x + m', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['google_id']})
    return fit


def summarize(name, d, subset_label='all'):
    fit = model2(d)
    return {
        'condition': name, 'subset': subset_label, 'n': len(d),
        'M_pos': int((d.m == 1).sum()), 'M_neu': int((d.m == 0).sum()), 'M_neg': int((d.m == -1).sum()),
        'five_star_scored_negative': int(((d.y == 5) & (d.m == -1)).sum()),
        'X_pos': int((d.x == 1).sum()), 'X_neg': int((d.x == -1).sum()),
        'r_M_Y': round(d.m.corr(d.y), 3), 'r_X_Y': round(d.x.corr(d.y), 3),
        'rho_M_Y': round(d.m.corr(d.y, method='spearman'), 3),
        'b_X': round(fit.params['x'], 3), 'p_X': round(fit.pvalues['x'], 4),
        'b_M': round(fit.params['m'], 3), 'p_M': round(fit.pvalues['m'], 4),
        'R2': round(fit.rsquared, 3),
    }


def main():
    translated = pd.read_csv(os.path.join(OUTPUT, 'reviews_translated.csv'))
    analysed = pd.read_csv(os.path.join(OUTPUT, 'reviews_preprocessed.csv'))
    df = translated.merge(analysed[KEY], on=KEY, how='inner').reset_index(drop=True)

    lexicon = load_lexicon(POS_LEXICON_FILE, NEG_LEXICON_FILE)
    keywords = load_service_keywords(SERVICE_KEYWORDS_FILE)
    prep = Preprocessor()

    def effective(target):
        """A substitution is effective if its target survives preprocessing and is scored or negates."""
        for w in target.split():
            if w in prep.stop:
                continue
            s = prep.stem(w)
            if s in NEGATION_WORDS or lexicon.get(s, 0.0) != 0.0:
                return True
        return False

    rows, scored = [], {}
    for name, lex in resources().items():
        pattern = (re.compile(r'\b(' + '|'.join(map(re.escape, lex)) + r')\b', re.IGNORECASE)
                   if lex else None)
        texts, n_subs, n_eff = [], [], []
        for t in df['review_text']:
            norm, targets = normalize(t, lex, pattern)
            texts.append(prep(norm))
            n_subs.append(len(targets))
            n_eff.append(sum(effective(x) for x in targets))
        d = pd.DataFrame({
            'google_id': df['google_id'], 'y': df['review_rating'].astype(float),
            'subs': n_subs, 'eff': n_eff, 'text': texts,
        })
        d['m'] = d['text'].map(lambda s: classify_sentiment(calculate_sentiment_score(s, lexicon)))
        d['x'] = d['text'].map(lambda s: score_absa_service_quality(s, keywords, lexicon))
        scored[name] = d

        row = summarize(name, d)
        row.update({
            'entries': len(lex), 'substitutions': int(d.subs.sum()),
            'effective_substitutions': int(d.eff.sum()),
            'reach_pct': round(100 * (d.subs > 0).mean(), 1),
            'effective_reach_pct': round(100 * (d.eff > 0).mean(), 1),
        })
        rows.append(row)

    # Sanity check: the 'proposed' condition must reproduce the main pipeline exactly.
    main_text = analysed['review_text'].tolist()
    mismatch = sum(a != b for a, b in zip(scored['proposed']['text'], main_text))
    print(f'Reproduction check (proposed vs reviews_preprocessed.csv): {mismatch} mismatching reviews')

    # Same comparison restricted to the reviews the proposed dictionary actually changed.
    touched = scored['proposed']['subs'] > 0
    for name in ('none', 'colloquial', 'proposed', 'combined'):
        rows.append(summarize(name, scored[name][touched], 'touched_by_proposed'))
        rows.append(summarize(name, scored[name][~touched], 'untouched_by_proposed'))

    # Paired cluster bootstrap (resampling restaurants) for differences in correlation with Y.
    rng = np.random.default_rng(42)
    groups = df['google_id'].to_numpy()
    ids = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in ids}
    draws = [np.concatenate([idx_by_group[g] for g in rng.choice(ids, size=len(ids))])
             for _ in range(2000)]
    y = scored['none']['y'].to_numpy()

    def r(v, idx):
        return np.corrcoef(v[idx], y[idx])[0, 1]

    boot = []
    for a, b in [('proposed', 'none'), ('proposed', 'colloquial'), ('combined', 'colloquial'),
                 ('run1', 'none'), ('run2', 'none'), ('run3', 'none')]:
        for var in ('m', 'x'):
            va, vb = scored[a][var].to_numpy(), scored[b][var].to_numpy()
            diffs = np.array([r(va, i) - r(vb, i) for i in draws])
            boot.append({'comparison': f'{a} - {b}', 'variable': var.upper(),
                         'diff_r_with_Y': round(r(va, np.arange(len(y))) - r(vb, np.arange(len(y))), 3),
                         'ci95_low': round(np.percentile(diffs, 2.5), 3),
                         'ci95_high': round(np.percentile(diffs, 97.5), 3),
                         'share_of_draws_above_0': round((diffs > 0).mean(), 3)})
    boot = pd.DataFrame(boot)
    boot.to_csv(os.path.join(OUTPUT, 'ablation_bootstrap.csv'), index=False)
    print(boot.to_string(index=False), '\n')

    # Per-review scores for every condition, used by the gold-set evaluation
    pred = df[KEY].copy()
    for name, d in scored.items():
        pred[f'm_{name}'] = d['m'].to_numpy()
        pred[f'x_{name}'] = d['x'].to_numpy()
        pred[f'mention_{name}'] = d['text'].map(
            lambda s: int(any(w in keywords for w in s.split()))).to_numpy()
        pred[f'subs_{name}'] = d['subs'].to_numpy()
        pred[f'eff_{name}'] = d['eff'].to_numpy()
    pred.to_csv(os.path.join(OUTPUT, 'ablation_predictions.csv'), index=False)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT, 'ablation_results.csv'), index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    cols = ['condition', 'subset', 'n', 'entries', 'reach_pct', 'effective_reach_pct', 'M_pos', 'M_neg',
            'five_star_scored_negative', 'X_pos', 'X_neg', 'r_M_Y', 'r_X_Y', 'b_X', 'p_X', 'b_M', 'R2']
    print(out[[c for c in cols if c in out]].to_string(index=False))


if __name__ == '__main__':
    main()
