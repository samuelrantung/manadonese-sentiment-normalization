"""
Effect of curating the stop-word list (Section 3.2; reviewer comment on the NLTK list).

Compares the pipeline's curated list (NLTK Indonesian minus the negators, five evaluative words
and the contrast markers) with the full NLTK list. With the full list the contrast markers are
removed before scoring, so service windows can no longer stop at them. Everything else is held
fixed: proposed dictionary, domain table, window r = 5, negation scope 2.

Report: data/output/stopword_effect.txt
Run from the pipeline root:  python experiments/stopword_effect.py
"""
import contextlib
import io
import os
import re
import sys

import pandas as pd
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_DIR = os.path.join(BASE_DIR, 'annotation')
sys.path.insert(0, os.path.join(BASE_DIR, 'steps'))
sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))

from ablation import INPUT, KEY, OUTPUT, Preprocessor, load_json, normalize   # noqa: E402
from step7_sentiment_scoring import (                                          # noqa: E402
    NEG_LEXICON_FILE, POS_LEXICON_FILE, SERVICE_KEYWORDS_FILE, calculate_sentiment_score,
    classify_sentiment, load_lexicon, load_service_keywords, score_absa_service_quality)

REPORT = os.path.join(OUTPUT, 'stopword_effect.txt')
POL = {'Positif': 1, 'Netral-campuran': 0, 'Negatif': -1}


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        lexicon = load_lexicon(POS_LEXICON_FILE, NEG_LEXICON_FILE)
        keywords = load_service_keywords(SERVICE_KEYWORDS_FILE)
    translated = pd.read_csv(os.path.join(OUTPUT, 'reviews_translated.csv'))
    analysed = pd.read_csv(os.path.join(OUTPUT, 'reviews_preprocessed.csv'))
    df = translated.merge(analysed[KEY], on=KEY, how='inner').reset_index(drop=True)
    proposed = load_json(os.path.join(INPUT, 'manado_slang_dict.json'))
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, proposed)) + r')\b', re.IGNORECASE)
    normalized = [normalize(t, proposed, pattern)[0] for t in df['review_text']]

    labels = pd.read_csv(os.path.join(ANNOT_DIR, 'merged_labels.csv'))
    key = pd.read_csv(os.path.join(ANNOT_DIR, 'gold_sample_key.csv'))[['review_id'] + KEY]
    gold = labels.merge(key, on='review_id')
    agree = lambda c: gold[f'{c}_A'].where(gold[f'{c}_A'] == gold[f'{c}_B'])  # noqa: E731
    gold['g_overall'] = agree('overall_pol').map(POL)
    gold['g_service'] = agree('service_pol').map(POL).where(agree('mention') == 'Ya')

    curated = Preprocessor()
    full = Preprocessor()
    full.stop = set(stopwords.words('indonesian'))
    y = df['review_rating'].astype(float)
    out = [f'STOP-WORD CURATION (n = {len(df):,}; proposed dictionary, domain table, r = 5, scope 2)\n',
           f'  curated list removes {len(curated.stop)} words; full NLTK list removes {len(full.stop)}\n',
           f'  {"list":8} {"r(M,Y)":>7} {"r(X,Y)":>7} {"5* scored neg":>14} {"gold overall":>13} {"gold service":>13}']
    for name, prep in (('curated', curated), ('full', full)):
        texts = [prep(t) for t in normalized]
        d = df[KEY].assign(
            m=[classify_sentiment(calculate_sentiment_score(s, lexicon)) for s in texts],
            x=[score_absa_service_quality(s, keywords, lexicon) for s in texts])
        if name == 'curated':
            assert texts == analysed['review_text'].fillna('').tolist(), 'curated list does not reproduce Step 5'
        g = gold.merge(d, on=KEY)
        o, s = g['g_overall'].notna(), g['g_service'].notna()
        out.append(f'  {name:8} {d.m.corr(y):7.3f} {d.x.corr(y):7.3f} {((y == 5) & (d.m == -1)).sum():14d} '
                   f'{(g.m[o] == g.g_overall[o]).mean():13.3f} {(g.x[s] == g.g_service[s]).mean():13.3f}')
    out.append(f'\n  gold overall n = {gold.g_overall.notna().sum()}, gold service n = {gold.g_service.notna().sum()} '
               '(agreed labels)')

    text = '\n'.join(out)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
