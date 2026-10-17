"""
Surface-form vs stem lookup of InSet weights (review comment 17).

Starts from reviews_normalized.csv, applies the Step 5 stopword list, and keeps every
token in both its surface and its stemmed form. Three lookups are compared:
  stem          : InSet weight of the stem (the pipeline's choice)
  surface_first : InSet weight of the surface form if listed, otherwise of the stem
  either_max    : whichever of the two has the larger absolute weight
Domain overrides apply in all three (by stem, then by surface form).
Service windows are always located on stems, as in Step 7.

Results: data/output/lookup_comparison.csv
Run from the pipeline root:  python experiments/lookup_comparison.py
"""
import contextlib
import io
import os
import sys

import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'steps'))

import step7_sentiment_scoring as s7                                     # noqa: E402
from step5_text_preprocessing import build_stopwords                     # noqa: E402
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory              # noqa: E402

OUTPUT = os.path.join(BASE_DIR, 'data', 'output')


def load_weights():
    """Raw InSet (summed) and the domain overrides, kept separate."""
    with contextlib.redirect_stderr(io.StringIO()):
        saved = s7.DOMAIN_OVERRIDES_FILE
        s7.DOMAIN_OVERRIDES_FILE = os.path.join(OUTPUT, '__none__')
        inset = s7.load_lexicon(s7.POS_LEXICON_FILE, s7.NEG_LEXICON_FILE)
        s7.DOMAIN_OVERRIDES_FILE = saved
    ov = pd.read_csv(saved)
    return inset, dict(zip(ov['word'].str.strip().str.lower(), ov['new_weight'].astype(float)))


def weight(surface, stem, inset, ov, mode):
    if stem in ov:
        return ov[stem]
    if surface in ov:
        return ov[surface]
    s_w, t_w = inset.get(surface), inset.get(stem, 0.0)
    if mode == 'stem':
        return t_w
    if mode == 'surface_first':
        return s_w if s_w is not None else t_w
    return max((s_w or 0.0, t_w), key=abs)       # either_max


def score(stems, weights, positions):
    total = 0.0
    for i in positions:
        if stems[i] in s7.NEGATION_WORDS or weights[i] == 0.0:
            continue
        neg = any(stems[i - k] in s7.NEGATION_WORDS for k in (1, 2) if i - k >= 0)
        total += -weights[i] if neg else weights[i]
    return s7.classify_sentiment(total)


def main():
    df = pd.read_csv(os.path.join(OUTPUT, 'reviews_normalized.csv'))
    stop = build_stopwords()
    stemmer = StemmerFactory().create_stemmer()
    cache = {}
    inset, ov = load_weights()
    keywords = s7.load_service_keywords(s7.SERVICE_KEYWORDS_FILE)

    docs = []
    for t in df['review_text'].fillna(''):
        pairs = []
        for w in str(t).split():
            if w in stop:
                continue
            s = cache.setdefault(w, stemmer.stem(w))
            if s:
                pairs.append((w, s))
        docs.append(pairs)
    keep = [bool(p) for p in docs]
    df, docs = df[keep].reset_index(drop=True), [p for p in docs if p]

    tokens = [p for d in docs for p in d]
    match = {
        'surface': sum(w in inset for w, _ in tokens) / len(tokens),
        'stem': sum(s in inset for _, s in tokens) / len(tokens),
        'either': sum((w in inset) or (s in inset) for w, s in tokens) / len(tokens),
    }
    print('InSet match rate over %d tokens: ' % len(tokens)
          + ', '.join(f'{k} {v:.1%}' for k, v in match.items()))

    rows = []
    for mode in ('stem', 'surface_first', 'either_max'):
        m, x = [], []
        for pairs in docs:
            stems = [s for _, s in pairs]
            w = [weight(a, b, inset, ov, mode) for a, b in pairs]
            m.append(score(stems, w, range(len(stems))))
            x.append(score(stems, w, s7.service_window_positions(stems, keywords)))
        d = pd.DataFrame({'y': df['review_rating'].astype(float), 'g': df['google_id'], 'm': m, 'x': x})
        fit = smf.ols('y ~ x + m', d).fit(cov_type='cluster', cov_kwds={'groups': d['g']})
        rows.append({'lookup': mode, 'n': len(d),
                     'M_pos': int((d.m == 1).sum()), 'M_neg': int((d.m == -1).sum()),
                     'five_star_scored_negative': int(((d.y == 5) & (d.m == -1)).sum()),
                     'r_M_Y': round(d.m.corr(d.y), 3), 'r_X_Y': round(d.x.corr(d.y), 3),
                     'R2': round(fit.rsquared, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT, 'lookup_comparison.csv'), index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
