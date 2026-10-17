"""
Sensitivity of the scores to the window radius and the negation scope (review comment B-D22).

Grid: window radius r in {2, 3, 5, 7, full review} x negation scope in {1, 2, 3 tokens, clause}.
'clause' negates every scored token after a negator up to the next contrast marker
(the corpus has no punctuation after translation, so contrast markers stand in for clause
boundaries). All other rules are those of Step 7: negators are not scored, windows are the
union around service keywords and stop at contrast markers. The pipeline setting is r = 5, scope 2.

Optional: pass a merged_labels.csv to add gold-set accuracy (agreed or adjudicated labels).

Results: data/output/sensitivity.csv
Run from the pipeline root:  python experiments/sensitivity.py [annotation/merged_labels.csv]
"""
import contextlib
import io
import os
import sys

import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'steps'))
sys.path.insert(0, os.path.join(BASE_DIR, 'annotation'))

import step7_sentiment_scoring as s7                                     # noqa: E402

OUTPUT = os.path.join(BASE_DIR, 'data', 'output')
KEY = ['review_author', 'google_id', 'review_date']
POL = {'Positif': 1, 'Netral-campuran': 0, 'Negatif': -1}
RADII = [2, 3, 5, 7, 'full']
SCOPES = [1, 2, 3, 'clause']


def token_score(words, i, lexicon, scope):
    if words[i] in s7.NEGATION_WORDS:
        return 0.0
    wt = lexicon.get(words[i], 0.0)
    if wt == 0.0:
        return 0.0
    if scope == 'clause':
        neg = False
        for j in range(i - 1, -1, -1):
            if words[j] in s7.CONTRAST_WORDS:
                break
            if words[j] in s7.NEGATION_WORDS:
                neg = True
                break
    else:
        neg = any(words[i - k] in s7.NEGATION_WORDS for k in range(1, scope + 1) if i - k >= 0)
    return -wt if neg else wt


def score(text, keywords, lexicon, radius, scope):
    words = text.split()
    contrib = [token_score(words, i, lexicon, scope) for i in range(len(words))]
    if radius == 'full':
        window = range(len(words)) if any(w in keywords for w in words) else []
    else:
        window = s7.service_window_positions(words, keywords, window=radius)
    return s7.classify_sentiment(sum(contrib)), s7.classify_sentiment(sum(contrib[j] for j in window))


def main():
    df = pd.read_csv(os.path.join(OUTPUT, 'reviews_preprocessed.csv'))
    df['review_text'] = df['review_text'].fillna('')
    with contextlib.redirect_stderr(io.StringIO()):
        lexicon = s7.load_lexicon(s7.POS_LEXICON_FILE, s7.NEG_LEXICON_FILE)
        keywords = s7.load_service_keywords(s7.SERVICE_KEYWORDS_FILE)

    gold = None
    if len(sys.argv) > 1:
        from evaluate_gold import gold as gold_label
        labels = pd.read_csv(sys.argv[1])
        key = pd.read_csv(os.path.join(BASE_DIR, 'annotation', 'gold_sample_key.csv'))[['review_id'] + KEY]
        labels = labels.merge(key, on='review_id')
        labels['g_mention'] = gold_label(labels, 'mention').map({'Ya': 1, 'Tidak': 0})
        labels['g_service'] = gold_label(labels, 'service_pol').map(POL)
        labels['g_overall'] = gold_label(labels, 'overall_pol').map(POL)
        gold = labels[KEY + ['g_mention', 'g_service', 'g_overall']]

    rows = []
    for radius in RADII:
        for scope in SCOPES:
            ms, xs = zip(*(score(t, keywords, lexicon, radius, scope) for t in df['review_text']))
            d = df[KEY + ['review_rating']].copy()
            d['m'], d['x'], d['y'] = ms, xs, d['review_rating'].astype(float)
            fit = smf.ols('y ~ x + m', d).fit(cov_type='cluster', cov_kwds={'groups': d['google_id']})
            row = {'radius': radius, 'neg_scope': scope,
                   'pipeline_setting': radius == 5 and scope == 2,
                   'r_M_Y': round(d.m.corr(d.y), 3), 'r_X_Y': round(d.x.corr(d.y), 3),
                   'b_X': round(fit.params['x'], 3), 'p_X': round(fit.pvalues['x'], 4),
                   'R2': round(fit.rsquared, 3)}
            if gold is not None:
                g = gold.merge(d, on=KEY)
                s = g[(g.g_mention == 1)].dropna(subset=['g_service'])
                o = g.dropna(subset=['g_overall'])
                row.update({'gold_service_acc': round((s.x == s.g_service).mean(), 3), 'n_service': len(s),
                            'gold_overall_acc': round((o.m == o.g_overall).mean(), 3), 'n_overall': len(o)})
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT, 'sensitivity.csv'), index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
