"""
Supplementary gold-set statistics reported in Sections 4.1 and 4.3.

  1. Class distribution of the agreed gold labels and the majority-class baseline, as sampled and
     reweighted to the corpus rating distribution (the gold sample is stratified by rating)
  2. Correct predictions per gold class (service: lexicon vs hybrid; overall: lexicon vs transformer)
  3. Share of Indonesian-language gold reviews that contain Manadonese vocabulary, as sampled and
     reweighted to the corpus rating distribution (the gold sample is stratified by rating)
  4. Service terms that trigger false service detections (gold: no mention; lexicon: mention)

Usage (from the pipeline root):
    python annotation/gold_details.py
Writes annotation/gold_details.txt
"""
import os
from collections import Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_DIR = os.path.join(BASE_DIR, 'annotation')
OUTPUT = os.path.join(BASE_DIR, 'data', 'output')
KEY = ['review_author', 'google_id', 'review_date']
POL = {'Positif': 1, 'Netral-campuran': 0, 'Negatif': -1}
NAME = {-1: 'negative', 0: 'neutral', 1: 'positive'}
SERVICE_KEYWORDS_FILE = os.path.join(BASE_DIR, 'data', 'input', 'service_keywords.txt')


def main():
    labels = pd.read_csv(os.path.join(ANNOT_DIR, 'merged_labels.csv'))
    key = pd.read_csv(os.path.join(ANNOT_DIR, 'gold_sample_key.csv'))[['review_id', 'review_rating'] + KEY]
    abl = pd.read_csv(os.path.join(OUTPUT, 'ablation_predictions.csv'))
    bert = pd.read_csv(os.path.join(OUTPUT, 'indobert_predictions.csv'))[KEY + ['bert_normalized', 'x_hybrid']]
    prep = pd.read_csv(os.path.join(OUTPUT, 'reviews_preprocessed.csv'))
    d = (labels.merge(key, on='review_id').merge(abl, on=KEY).merge(bert, on=KEY)
         .merge(prep[KEY + ['review_text']], on=KEY))
    assert len(d) == len(labels)

    agree = lambda c: d[f'{c}_A'].where(d[f'{c}_A'] == d[f'{c}_B'])  # noqa: E731
    g_mention = agree('mention').map({'Ya': 1, 'Tidak': 0})
    g_service = agree('service_pol').map(POL)
    g_overall = agree('overall_pol').map(POL)
    s = (g_mention == 1) & g_service.notna()
    o = g_overall.notna()
    corpus = prep['review_rating'].value_counts()
    w = d['review_rating'].map(corpus / d['review_rating'].value_counts())

    out = []
    p = out.append
    p('GOLD-SET DETAILS (agreed labels unless stated)\n')
    p('1. CLASS DISTRIBUTION AND MAJORITY-CLASS BASELINE')
    for task, mask, gold in (('service polarity', s, g_service), ('overall polarity', o, g_overall)):
        c = gold[mask].value_counts()
        top = c.idxmax()
        p(f'  {task} (n = {mask.sum()}): ' + ', '.join(f'{NAME[k]} {c.get(k, 0)}' for k in (-1, 0, 1))
          + f'  -> always predicting {NAME[top]}: accuracy {c[top] / mask.sum():.3f}')
        cw = w[mask].groupby(gold[mask]).sum() / w[mask].sum()
        topw = cw.idxmax()
        p(f'    reweighted to the corpus rating mix: ' + ', '.join(f'{NAME[k]} {cw.get(k, 0):.3f}' for k in (-1, 0, 1))
          + f'  -> always predicting {NAME[topw]}: accuracy {cw[topw]:.3f}')

    p('\n2. CORRECT PREDICTIONS PER GOLD CLASS')
    for task, mask, gold, scorers in (
            ('service polarity', s, g_service, (('lexicon X, proposed', 'x_proposed'), ('hybrid X', 'x_hybrid'))),
            ('overall polarity', o, g_overall, (('lexicon M, proposed', 'm_proposed'), ('transformer, proposed', 'bert_normalized')))):
        p(f'  {task}:')
        for label, col in scorers:
            parts = []
            for k in (-1, 0, 1):
                m = mask & (gold == k)
                parts.append(f'{NAME[k]} {(d.loc[m, col] == k).sum()}/{m.sum()}')
            p(f'    {label:24} ' + ', '.join(parts))

    p('\n3. MANADONESE VOCABULARY IN INDONESIAN-LANGUAGE GOLD REVIEWS')
    corpus = prep['review_rating'].value_counts()
    sample = d['review_rating'].value_counts()
    ind = d[d['language'] == 'Indonesia/Manado']
    for label, flag in (('both annotators', (ind.manado_A == 'Ya') & (ind.manado_B == 'Ya')),
                        ('either annotator', (ind.manado_A == 'Ya') | (ind.manado_B == 'Ya'))):
        t = ind.assign(f=flag.astype(int)).groupby('review_rating')['f'].agg(['size', 'sum'])
        # Post-stratification: each rating stratum is scaled up to its size in the corpus.
        w = corpus.reindex(t.index) / sample.reindex(t.index)
        weighted = (w * t['sum']).sum() / (w * t['size']).sum()
        by = ', '.join(f'{r}* {t.loc[r, "sum"]}/{t.loc[r, "size"]}' for r in t.index)
        p(f'  {label:16}: sample {flag.sum()}/{len(ind)} ({flag.mean():.1%}); '
          f'reweighted to the corpus rating mix {weighted:.1%}  [{by}]')

    p('\n4. FALSE SERVICE DETECTIONS (gold: no mention; lexicon window, proposed: mention)')
    with open(SERVICE_KEYWORDS_FILE, encoding='utf-8') as f:
        keywords = {line.strip().lower() for line in f if line.strip()}
    fp = d[(g_mention == 0) & (d['mention_proposed'] == 1)]
    triggers = [{w for w in str(t).split() if w in keywords} for t in fp['review_text']]
    p(f'  False detections: {len(fp)}')
    p('  Reviews containing each trigger: ' + ', '.join(
        f'{w} {n}' for w, n in Counter(w for t in triggers for w in t).most_common(10)))
    four = {'pesan', 'sedia', 'saji', 'jam'}
    p(f'  Triggered only by pesan/sedia/saji/jam: {sum(1 for t in triggers if t <= four)}')

    text = '\n'.join(out)
    with open(os.path.join(ANNOT_DIR, 'gold_details.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
