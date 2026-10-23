"""
Descriptive analyses requested in review (corpus, translation, dictionary, lexicon).

  A. Corpus: reviews per city and per restaurant; star-rating distribution        (B-D28)
  B. Translation: reviews changed by translation; estimated language mix and its
     validation (overlap distribution, threshold sensitivity, agreement with the
     annotators' language labels); Manadonese markers before and after
     translation; the false friend 'kita'                                          (B-D8, D29, D31)
  C. Dictionary: share of substitutions that are effective; token-level InSet
     out-of-vocabulary rate before and after normalization; provenance of 'tidak' (B-D14, D16, D36)
  D. Service lexicon: corpus frequency of the terms added beyond the top-200 list;
     reviews whose X was triggered only by terima/kasih in the submitted version;
     anchor weights before and after the domain table                             (B-D18, D19)
  E. Lexicon: top-20 positive and negative contributors before and after
     the domain table                                                              (B-D25)
  F. Multiword expressions named by the reviewer: frequency and current handling   (B-D15)

Report: data/output/descriptives.txt
Run from the pipeline root:  python experiments/descriptives.py
"""
import contextlib
import io
import os
import re
import sys
from collections import Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'steps'))
sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))

import step7_sentiment_scoring as s7                                     # noqa: E402
from ablation import Preprocessor, load_json, normalize                  # noqa: E402

INPUT = os.path.join(BASE_DIR, 'data', 'input')
OUTPUT = os.path.join(BASE_DIR, 'data', 'output')
ANNOTATION = os.path.join(BASE_DIR, 'annotation')
# Outputs of the submitted version, kept unchanged for comparison
SUBMITTED = os.path.join(BASE_DIR, 'data', 'submitted_version')
KEY = ['review_author', 'google_id', 'review_date']
MANADO_MARKERS = ['nyanda', 'nda', 'ndak', 'torang', 'dorang', 'ngana', 'ngoni', 'mar', 'kong',
                  'dang', 'jo', 'sadap', 'makang', 'dpe', 'depe', 'pigi', 'kita', 'so', 'mo']
MWES = ['bukan main', 'tidak hanya', 'kurang lebih', 'tidak kalah', 'terima kasih', 'salah satu']


def words(text):
    return re.findall(r'[a-z]+', str(text).lower())


def lexicons():
    with contextlib.redirect_stderr(io.StringIO()):
        final = s7.load_lexicon(s7.POS_LEXICON_FILE, s7.NEG_LEXICON_FILE)
        saved = s7.DOMAIN_OVERRIDES_FILE
        s7.DOMAIN_OVERRIDES_FILE = os.path.join(OUTPUT, '__none__')
        raw = s7.load_lexicon(s7.POS_LEXICON_FILE, s7.NEG_LEXICON_FILE)
        s7.DOMAIN_OVERRIDES_FILE = saved
    return raw, final


def contributions(texts, lexicon, score_negators):
    """Summed signed contribution of each word to M across the corpus."""
    total = Counter()
    for t in texts:
        w = t.split()
        for i, word in enumerate(w):
            if not score_negators and word in s7.NEGATION_WORDS:
                continue
            wt = lexicon.get(word, 0.0)
            if wt == 0.0:
                continue
            neg = any(w[i - k] in s7.NEGATION_WORDS for k in (1, 2) if i - k >= 0)
            total[word + (' (negated)' if neg else '')] += -wt if neg else wt
    return total


def main():
    out = []
    p = out.append
    cleaned = pd.read_csv(os.path.join(OUTPUT, 'reviews_cleaned.csv'))
    translated = pd.read_csv(os.path.join(OUTPUT, 'reviews_translated.csv'))
    normalized = pd.read_csv(os.path.join(OUTPUT, 'reviews_normalized.csv'))
    pre = pd.read_csv(os.path.join(OUTPUT, 'reviews_preprocessed.csv'))
    raw_lex, lex = lexicons()
    keywords = s7.load_service_keywords(s7.SERVICE_KEYWORDS_FILE)

    # ---------- A. corpus ----------
    p('A. CORPUS (analysed sample, n = %d)' % len(pre))
    # The raw data has no address field, so each restaurant's city is recorded by hand in
    # restaurant_city.csv (filled from the restaurant name where it names the city or a known area).
    cities = pd.read_csv(os.path.join(INPUT, 'restaurant_city.csv'))
    pre['city'] = pre['google_id'].map(dict(zip(cities['google_id'], cities['city'].fillna(''))))
    pre['city'] = pre['city'].replace('', 'not yet assigned')
    by_city = pre.groupby('city').agg(reviews=('google_id', 'size'), restaurants=('google_id', 'nunique'))
    p(by_city.to_string())
    per_rest = pre.groupby('google_id').size()
    p(f'Reviews per restaurant: median {per_rest.median():.0f}, min {per_rest.min()}, max {per_rest.max()}, '
      f'restaurants {len(per_rest)}')
    p('Star ratings: ' + ', '.join(f'{k}* {v}' for k, v in pre['review_rating'].value_counts().sort_index().items())
      + f'; mean {pre.review_rating.mean():.2f}, SD {pre.review_rating.std():.2f}')

    # ---------- B. translation ----------
    p('\nB. TRANSLATION (1,098 cleaned reviews)')
    log = open(os.path.join(OUTPUT, 'step3_translation.log'), encoding='utf-8').read()
    p(f'  Rows still untranslated at the end of Step 3: '
      f'{0 if "All rows translated" in log else "see log"} (failures were retried until all succeeded)')
    tok_o = cleaned['review_text'].map(lambda t: set(words(t)))
    tok_t = translated['review_text'].map(lambda t: set(words(t)))
    overlap = [len(a & b) / len(a) if a else 1.0 for a, b in zip(tok_o, tok_t)]
    changed = sum(o < 0.5 for o in overlap)
    p(f'  Reviews whose words changed by more than half in translation (i.e. translated from another '
      f'language): {changed} ({changed / len(cleaned):.1%})')
    hangul = cleaned['review_text'].astype(str).str.contains(r'[가-힯]').sum()
    p(f'  Of these, Korean script: {hangul}. Gold-sample hand check: 72 of 300 reviews (24.0%) were '
      f'non-Indonesian (51 English, 21 other).')
    # Validation of the estimate (B-D8): the detected language was not recorded, so the
    # word-overlap rule is checked against its own distribution and the annotators' labels
    p(f'  Word overlap after translation: under 10% in {sum(o < 0.1 for o in overlap)} reviews, '
      f'at least 90% in {sum(o >= 0.9 for o in overlap)}, in between in '
      f'{sum(0.1 <= o < 0.9 for o in overlap)}')
    p('  Threshold sensitivity (reviews with overlap below t): ' + ', '.join(
        f't = {t:.1f}: {sum(o < t for o in overlap)}' for t in (0.3, 0.4, 0.5, 0.6, 0.7)))
    gold = pd.read_csv(os.path.join(ANNOTATION, 'gold_sample_key.csv')).merge(
        pd.read_csv(os.path.join(ANNOTATION, 'merged_labels.csv'))[['review_id', 'language']],
        on='review_id')
    g_o = gold['original_text'].map(lambda t: set(words(t)))
    g_t = gold['translated_text'].map(lambda t: set(words(t)))
    flagged = pd.Series([(len(a & b) / len(a) if a else 1.0) < 0.5 for a, b in zip(g_o, g_t)])
    foreign = gold['language'] != 'Indonesia/Manado'
    p(f'  Gold sample, rule (t = 0.5) vs annotators\' language labels: flags {(flagged & foreign).sum()} '
      f'of {foreign.sum()} non-Indonesian reviews and {(flagged & ~foreign).sum()} of '
      f'{(~foreign).sum()} Indonesian/Manadonese reviews')
    p('  Manadonese markers: occurrences in the original text -> occurrences surviving translation')
    for m in MANADO_MARKERS:
        a = sum(Counter(words(t))[m] for t in cleaned['review_text'])
        b = sum(Counter(words(t))[m] for t in translated['review_text'])
        if a:
            p(f'    {m:8} {a:4} -> {b:4}')
    p(f"  'kita' (Manadonese 'I', standard 'we') is on the NLTK stop-word list: "
      f"{'kita' in Preprocessor().stop} -> it never reaches the scorer.")

    # ---------- C. dictionary ----------
    p('\nC. DICTIONARY (proposed, 109 entries)')
    prop = load_json(os.path.join(INPUT, 'manado_slang_dict.json'))
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, prop)) + r')\b', re.IGNORECASE)
    prep = Preprocessor()
    target_counts = Counter()
    for t in translated['review_text']:
        _, targets = normalize(t, prop, pattern)
        target_counts.update(targets)
    n_subs = sum(target_counts.values())

    def effect(target):
        ws = [w for w in target.split() if w not in prep.stop]
        if not ws:
            return 'removed as stop-word'
        st = [prep.stem(w) for w in ws]
        if any(s in s7.NEGATION_WORDS for s in st):
            return 'negator (reverses next words)'
        if any(lex.get(s, 0.0) != 0.0 for s in st):
            return 'carries a sentiment weight'
        return 'kept but unweighted'
    eff = Counter()
    for tgt, c in target_counts.items():
        eff[effect(tgt)] += c
    p(f'  Substitutions: {n_subs:,}')
    for k, v in eff.most_common():
        p(f'    {k:32} {v:5} ({v / n_subs:.1%})')
    p('  Most frequent targets: ' + ', '.join(f'{t} {c}' for t, c in target_counts.most_common(12)))

    def oov(texts):
        toks = [w for t in texts for w in prep(t).split()]
        return sum(1 for w in toks if w not in lex and w not in s7.NEGATION_WORDS) / len(toks), len(toks)
    before = [normalize(t, {}, None)[0] for t in translated['review_text']]
    after = [normalize(t, prop, pattern)[0] for t in translated['review_text']]
    (o1, n1), (o2, n2) = oov(before), oov(after)
    p(f'  Token-level out-of-vocabulary rate against InSet (after stop-words and stemming): '
      f'{o1:.1%} of {n1:,} tokens before normalization -> {o2:.1%} of {n2:,} after')
    t_before = sum(Counter(t.split())['tidak'] for t in before)
    t_after = sum(Counter(t.split())['tidak'] for t in after)
    p(f"  'tidak' tokens: {t_before} already in the translated text, {t_after - t_before} added by the "
      f'dictionary ({t_after} total before stop-word removal)')

    # ---------- D. service lexicon ----------
    p('\nD. SERVICE LEXICON')
    top200 = set(pd.read_csv(os.path.join(OUTPUT, 'top_words.csv')).iloc[:, 0].astype(str))
    sub_counts = Counter(w for t in pre['review_text'].fillna('') for w in t.split())
    original_terms = [l.strip() for l in open(os.path.join(SUBMITTED, 'service_keywords.txt'))
                      if l.strip()]
    expansions = [k for k in original_terms if k not in top200]
    p(f'  Terms in the submitted lexicon: {len(original_terms)}; from the top-200 list: '
      f'{len(original_terms) - len(expansions)}; added beyond it: {len(expansions)}')
    p('  Corpus frequency of the added terms: ' + ', '.join(f'{k} {sub_counts[k]}' for k in sorted(
        expansions, key=lambda k: -sub_counts[k])))
    p(f'  Added terms that never occur in the corpus: {sum(sub_counts[k] == 0 for k in expansions)}')

    sub_pre = pd.read_csv(os.path.join(SUBMITTED, 'reviews_preprocessed.csv'))
    sub_sc = pd.read_csv(os.path.join(SUBMITTED, 'reviews_scored.csv'))
    only_tk = 0
    for t, x in zip(sub_pre['review_text'].fillna(''), sub_sc['service_quality_X']):
        hits = {w for w in t.split() if w in set(original_terms)}
        if hits and hits <= {'terima', 'kasih'} and x != 0:
            only_tk += 1
    p(f'  Submitted version: reviews with X != 0 triggered only by terima/kasih: {only_tk}')
    p('  Anchor weights (InSet net -> final): ' + ', '.join(
        f'{k} {raw_lex.get(k, 0):+.0f}->{lex.get(k, 0):+.0f}' for k in sorted(original_terms)
        if raw_lex.get(k, 0) != 0 or lex.get(k, 0) != 0))

    # ---------- E. top contributors ----------
    p('\nE. TOP CONTRIBUTORS TO M (summed signed weight over the corpus)')
    texts = pre['review_text'].fillna('').tolist()
    for label, lx, scored_neg in (('Before (raw InSet, negators scored)', raw_lex, True),
                                  ('After (domain table, negators not scored)', lex, False)):
        c = contributions(texts, lx, scored_neg)
        p(f'  {label}')
        p('    negative: ' + ', '.join(f'{w} {v:.0f}' for w, v in sorted(c.items(), key=lambda x: x[1])[:20]))
        p('    positive: ' + ', '.join(f'{w} {v:.0f}' for w, v in sorted(c.items(), key=lambda x: -x[1])[:20]))

    # ---------- F. multiword expressions ----------
    p('\nF. MULTIWORD EXPRESSIONS (translated text, before Step 4 and stop-word removal)')
    norm_texts = translated['review_text'].fillna('').astype(str).str.lower()
    for m in MWES:
        n = norm_texts.str.count(r'\b' + m.replace(' ', r'\s+') + r'\b').sum()
        p(f'  {m:14} {n:4}')
    p("  ('salah satu' is rewritten to 'satu' in Step 4; 'terima'/'kasih' are no longer service anchors)")

    text = '\n'.join(out)
    with open(os.path.join(OUTPUT, 'descriptives.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
