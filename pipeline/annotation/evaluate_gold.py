"""
Evaluates every scorer against the human gold labels, with cluster-bootstrap confidence intervals.

Gold labels come from merged_labels.csv (written by check_annotations.py). Three gold modes:
  agreed : the label both annotators agree on; items where they disagree are left out (primary)
  A, B   : one annotator's labels for every item (robustness: shows that conclusions do not
           depend on how disagreements are handled)

Tasks
  1. Service mention detection : precision, recall, F1 for 'mentions service'
  2. Service polarity          : accuracy and macro-F1 on reviews whose gold label mentions
                                 service (a predicted 0 counts as neutral)
  3. Overall polarity          : accuracy and macro-F1
Each task is reported for all reviews and for Indonesian/Manadonese reviews only.

For service polarity, the transformer's whole-review polarity is also scored as a service polarity
where the lexicon finds a service term (0 elsewhere, as for the hybrid), to test whether classifying
the service window adds anything over classifying the whole review.

Accuracy is also reported reweighted to the corpus's rating distribution (acc_weighted): the sample
is stratified by rating and over-represents low ratings, so each gold review is weighted by the
number of corpus reviews with its rating divided by the number of sampled reviews with that rating.

Uncertainty: 95% percentile intervals from a bootstrap that resamples restaurants (2,000 draws,
seed 42), because reviews of one restaurant are not independent. Differences between scorers are
paired: both scorers are evaluated on the same resample.

Usage (from the pipeline root):
    python annotation/evaluate_gold.py annotation/merged_labels.csv [agreed|A|B]
Writes gold_evaluation_<mode>.csv and gold_differences_<mode>.csv next to the labels file.
"""
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_DIR = os.path.join(BASE_DIR, 'annotation')
OUTPUT = os.path.join(BASE_DIR, 'data', 'output')
KEY = ['review_author', 'google_id', 'review_date']
POL = {'Positif': 1, 'Netral-campuran': 0, 'Negatif': -1}
N_BOOT, SEED = 2000, 42

MENTION = {'lexicon window, no normalization': 'mention_none',
           'lexicon window, proposed dictionary': 'mention_proposed',
           'surface window (hybrid)': 'hybrid_mention'}
SERVICE = {'lexicon X, no normalization': 'x_none',
           'lexicon X, colloquial lexicon': 'x_colloquial',
           'lexicon X, proposed dictionary': 'x_proposed',
           'lexicon X, combined': 'x_combined',
           'lexicon X, re-run 1': 'x_run1', 'lexicon X, re-run 2': 'x_run2', 'lexicon X, re-run 3': 'x_run3',
           'hybrid X (window + transformer)': 'x_hybrid',
           'transformer, whole review, no normalization': 'x_bert_whole_none',
           'transformer, whole review, proposed dictionary': 'x_bert_whole'}
OVERALL = {'lexicon M, no normalization': 'm_none',
           'lexicon M, colloquial lexicon': 'm_colloquial',
           'lexicon M, proposed dictionary': 'm_proposed',
           'lexicon M, combined': 'm_combined',
           'lexicon M, re-run 1': 'm_run1', 'lexicon M, re-run 2': 'm_run2', 'lexicon M, re-run 3': 'm_run3',
           'transformer, no normalization': 'bert_translated',
           'transformer, proposed dictionary': 'bert_normalized'}
# Paired comparisons reported with intervals: (task, label, scorer a, scorer b) -> a minus b
DIFFS = [
    ('service', 'hybrid − lexicon (proposed)', 'x_hybrid', 'x_proposed'),
    ('service', 'lexicon: proposed − none', 'x_proposed', 'x_none'),
    ('service', 'lexicon: proposed − colloquial', 'x_proposed', 'x_colloquial'),
    ('overall', 'transformer − lexicon (both proposed)', 'bert_normalized', 'm_proposed'),
    ('overall', 'lexicon: proposed − none', 'm_proposed', 'm_none'),
    ('overall', 'lexicon: proposed − colloquial', 'm_proposed', 'm_colloquial'),
    ('overall', 'lexicon: combined − colloquial', 'm_combined', 'm_colloquial'),
    ('overall', 'transformer: proposed − none', 'bert_normalized', 'bert_translated'),
    ('service', 'hybrid − transformer whole review (both proposed)', 'x_hybrid', 'x_bert_whole'),
    ('service', 'transformer whole review − lexicon (proposed)', 'x_bert_whole', 'x_proposed'),
]


def gold(labels, col, mode='agreed'):
    if f'{col}_gold' in labels:
        return labels[f'{col}_gold']
    if mode in ('A', 'B'):
        return labels[f'{col}_{mode}']
    a, b = labels[f'{col}_A'], labels[f'{col}_B']
    return a.where(a == b)


def prf(pred, true):
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if (p + r) else np.nan
    return p, r, f


def accuracy(pred, true):
    return np.mean(pred == true) if len(true) else np.nan


def waccuracy(pred, true, w):
    return float(np.average(pred == true, weights=w)) if len(true) else np.nan


def macro_f1(pred, true):
    f1s = []
    for c in sorted(set(true.tolist()) | set(pred.tolist())):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1s)) if f1s else np.nan


def cluster_draws(groups, rng):
    ids = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ids}
    return [np.concatenate([pos[g] for g in rng.choice(ids, size=len(ids))]) for _ in range(N_BOOT)]


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return (round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)) if len(v) else (np.nan, np.nan)


def main():
    labels_path = sys.argv[1]
    gold_mode = sys.argv[2] if len(sys.argv) > 2 else 'agreed'
    labels = pd.read_csv(labels_path)
    key = pd.read_csv(os.path.join(ANNOT_DIR, 'gold_sample_key.csv'))[['review_id', 'review_rating'] + KEY]
    abl = pd.read_csv(os.path.join(OUTPUT, 'ablation_predictions.csv'))
    bert = pd.read_csv(os.path.join(OUTPUT, 'indobert_predictions.csv'))[
        KEY + ['bert_translated', 'bert_normalized', 'x_hybrid', 'service_span']]
    d = labels.merge(key, on='review_id').merge(abl, on=KEY).merge(bert, on=KEY)
    assert len(d) == len(labels), 'some gold reviews are missing from the predictions'

    d['g_mention'] = gold(d, 'mention', gold_mode).map({'Ya': 1, 'Tidak': 0})
    d['g_service'] = gold(d, 'service_pol', gold_mode).map(POL)
    d['g_overall'] = gold(d, 'overall_pol', gold_mode).map(POL)
    d['hybrid_mention'] = d['service_span'].fillna('').astype(str).str.len().gt(0).astype(int)
    d['x_bert_whole'] = d['bert_normalized'].where(d['mention_proposed'] == 1, 0)
    d['x_bert_whole_none'] = d['bert_translated'].where(d['mention_none'] == 1, 0)
    corpus = pd.read_csv(os.path.join(OUTPUT, 'reviews_preprocessed.csv'))['review_rating'].value_counts()
    d['w'] = d['review_rating'].map(corpus / d['review_rating'].value_counts())
    rng = np.random.default_rng(SEED)

    rows, diffs = [], []
    for subset, sd in (('all', d), ('Indonesian/Manadonese', d[d['language'] == 'Indonesia/Manado'])):
        sd = sd.reset_index(drop=True)
        m = sd.dropna(subset=['g_mention']).reset_index(drop=True)
        s = sd[sd['g_mention'] == 1].dropna(subset=['g_service']).reset_index(drop=True)
        o = sd.dropna(subset=['g_overall']).reset_index(drop=True)
        draws = {name: cluster_draws(t['google_id'].to_numpy(), rng) for name, t in
                 (('mention', m), ('service', s), ('overall', o))}

        true = m['g_mention'].to_numpy()
        for label, col in MENTION.items():
            pred = m[col].to_numpy()
            p, r, f = prf(pred, true)
            boot = [prf(pred[i], true[i])[2] for i in draws['mention']]
            lo, hi = ci(boot)
            rows.append({'task': '1 service mention', 'subset': subset, 'scorer': label, 'n': len(m),
                         'precision': round(p, 3), 'recall': round(r, 3), 'F1': round(f, 3),
                         'F1_ci_low': lo, 'F1_ci_high': hi})

        for task, table, cols in (('2 service polarity', s, SERVICE), ('3 overall polarity', o, OVERALL)):
            name = 'service' if task.startswith('2') else 'overall'
            true, w = table['g_' + name].to_numpy(), table['w'].to_numpy()
            for label, col in cols.items():
                pred = table[col].to_numpy()
                acc_b = [accuracy(pred[i], true[i]) for i in draws[name]]
                f1_b = [macro_f1(pred[i], true[i]) for i in draws[name]]
                accw_b = [waccuracy(pred[i], true[i], w[i]) for i in draws[name]]
                (al, ah), (fl, fh), (wl, wh) = ci(acc_b), ci(f1_b), ci(accw_b)
                rows.append({'task': task, 'subset': subset, 'scorer': label, 'n': len(table),
                             'accuracy': round(accuracy(pred, true), 3), 'acc_ci_low': al, 'acc_ci_high': ah,
                             'macro_F1': round(macro_f1(pred, true), 3), 'F1_ci_low': fl, 'F1_ci_high': fh,
                             'acc_weighted': round(waccuracy(pred, true, w), 3),
                             'accw_ci_low': wl, 'accw_ci_high': wh})

        for name, label, a, b in DIFFS:
            table = s if name == 'service' else o
            true, w = table['g_' + name].to_numpy(), table['w'].to_numpy()
            pa, pb = table[a].to_numpy(), table[b].to_numpy()
            point = accuracy(pa, true) - accuracy(pb, true)
            boot = np.array([accuracy(pa[i], true[i]) - accuracy(pb[i], true[i]) for i in draws[name]])
            bootw = [waccuracy(pa[i], true[i], w[i]) - waccuracy(pb[i], true[i], w[i]) for i in draws[name]]
            lo, hi = ci(boot)
            wlo, whi = ci(bootw)
            diffs.append({'task': name, 'subset': subset, 'comparison': label, 'n': len(table),
                          'accuracy_difference': round(point, 3), 'ci_low': lo, 'ci_high': hi,
                          'share_of_draws_above_0': round(float(np.mean(boot > 0)), 3),
                          'weighted_difference': round(waccuracy(pa, true, w) - waccuracy(pb, true, w), 3),
                          'w_ci_low': wlo, 'w_ci_high': whi})

    folder = os.path.dirname(os.path.abspath(labels_path))
    out, dif = pd.DataFrame(rows), pd.DataFrame(diffs)
    out.to_csv(os.path.join(folder, f'gold_evaluation_{gold_mode}.csv'), index=False)
    dif.to_csv(os.path.join(folder, f'gold_differences_{gold_mode}.csv'), index=False)

    mode = {'agreed': 'labels both annotators agree on', 'A': 'annotator A only', 'B': 'annotator B only'}[gold_mode]
    print(f'Gold labels: {mode}\n')
    pd.set_option('display.width', 220)
    for task, t in out.groupby('task'):
        print(task.upper())
        print(t.drop(columns='task').dropna(axis=1, how='all').to_string(index=False), '\n')
    print('PAIRED DIFFERENCES IN ACCURACY (95% cluster-bootstrap CI)')
    print(dif.to_string(index=False))


if __name__ == '__main__':
    main()
