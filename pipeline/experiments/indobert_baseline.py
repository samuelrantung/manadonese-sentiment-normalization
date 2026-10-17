"""
Non-lexicon baseline: a pretrained Indonesian transformer sentiment classifier.

Model: w11wo/indonesian-roberta-base-sentiment-classifier (Hugging Face), an Indonesian
RoBERTa fine-tuned on the IndoNLU SmSA sentence-sentiment dataset (positive/neutral/negative).
It is used zero-shot on this corpus (no fine-tuning). The resolved model commit is logged.

It scores the review text twice, with and without the dialect dictionary:
  translated : reviews_translated.csv (before normalization)
  normalized : reviews_normalized.csv (after the proposed dictionary)
Predictions are mapped to -1 / 0 / +1 and compared with the lexicon score M on the same
1,096 analysed reviews. Per-review predictions are saved for the gold-set evaluation.

It also runs a hybrid service-aspect scorer: the pipeline's service lexicon locates service
mentions (matched on Sastrawi stems, as in Step 7) and IndoBERT classifies only the words around
them (a surface window of +/-SURFACE_WINDOW words, stopping at contrast markers). Reviews
with no service mention get 0, as in the lexicon X.

Outputs: data/output/indobert_predictions.csv, data/output/indobert_summary.csv
Run from the pipeline root:  python experiments/indobert_baseline.py
"""
import os
import sys

import pandas as pd
import statsmodels.formula.api as smf
import torch
from huggingface_hub import model_info
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE_DIR, 'data', 'output')
MODEL = 'w11wo/indonesian-roberta-base-sentiment-classifier'
KEY = ['review_author', 'google_id', 'review_date']
POLARITY = {'positive': 1, 'neutral': 0, 'negative': -1}
# The lexicon window is +/-5 tokens AFTER stopword removal; about a third of surface words are
# stopwords, so +/-7 surface words covers roughly the same span.
SURFACE_WINDOW = 7

sys.path.insert(0, os.path.join(BASE_DIR, 'steps'))
from step7_sentiment_scoring import CONTRAST_WORDS, SERVICE_KEYWORDS_FILE, load_service_keywords  # noqa: E402
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # noqa: E402


def service_spans(texts, keywords):
    """Surface-text service windows for the hybrid scorer ('' when no service term)."""
    stemmer, cache = StemmerFactory().create_stemmer(), {}
    spans = []
    for text in texts:
        words = text.split()
        keep = set()
        for i, w in enumerate(words):
            if cache.setdefault(w, stemmer.stem(w)) not in keywords:
                continue
            keep.add(i)
            for step in (-1, 1):
                j = i + step
                while 0 <= j < len(words) and abs(j - i) <= SURFACE_WINDOW:
                    if words[j] in CONTRAST_WORDS:
                        break
                    keep.add(j)
                    j += step
        spans.append(' '.join(words[j] for j in sorted(keep)))
    return spans


def predict(texts, tokenizer, model, batch_size=32):
    labels = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                              max_length=256, return_tensors='pt')
            ids = model(**batch).logits.argmax(dim=-1).tolist()
            labels += [POLARITY[model.config.id2label[j].lower()] for j in ids]
    return labels


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    commit = model_info(MODEL).sha
    print(f'Model: {MODEL} @ {commit}; labels: {model.config.id2label}')

    analysed = pd.read_csv(os.path.join(OUTPUT, 'reviews_scored.csv'))
    base = pd.read_csv(os.path.join(OUTPUT, 'reviews_preprocessed.csv'))[KEY]
    base[['review_rating', 'm', 'x']] = analysed[['review_rating', 'customer_satisfaction_M',
                                                  'service_quality_X']].to_numpy()
    for name in ('translated', 'normalized'):
        src = pd.read_csv(os.path.join(OUTPUT, f'reviews_{name}.csv'))[KEY + ['review_text']]
        merged = base[KEY].merge(src, on=KEY, how='left')
        base[f'bert_{name}'] = predict(merged['review_text'].fillna('').astype(str).tolist(),
                                       tokenizer, model)

    # Hybrid service-aspect scorer on the normalized text
    norm = base[KEY].merge(pd.read_csv(os.path.join(OUTPUT, 'reviews_normalized.csv'))[KEY + ['review_text']],
                           on=KEY, how='left')['review_text'].fillna('').astype(str).tolist()
    spans = service_spans(norm, load_service_keywords(SERVICE_KEYWORDS_FILE))
    has_span = [bool(t) for t in spans]
    preds = iter(predict([t for t in spans if t], tokenizer, model))
    base['service_span'] = spans
    base['x_hybrid'] = [next(preds) if h else 0 for h in has_span]

    base.to_csv(os.path.join(OUTPUT, 'indobert_predictions.csv'), index=False)

    y, g = base['review_rating'].astype(float), base['google_id']
    rows = []
    for col, label in (('m', 'lexicon M (proposed pipeline)'),
                       ('bert_translated', 'IndoBERT, no normalization'),
                       ('bert_normalized', 'IndoBERT, with proposed dictionary')):
        v = base[col].astype(float)
        fit = smf.ols('y ~ v', pd.DataFrame({'y': y, 'v': v})).fit(
            cov_type='cluster', cov_kwds={'groups': g})
        rows.append({'scorer': label, 'pos': int((v == 1).sum()), 'neu': int((v == 0).sum()),
                     'neg': int((v == -1).sum()),
                     'five_star_scored_negative': int(((y == 5) & (v == -1)).sum()),
                     'r_with_Y': round(v.corr(y), 3), 'R2': round(fit.rsquared, 3),
                     'agreement_with_M': round((v == base['m']).mean(), 3)})
    for col, label in (('x', 'lexicon X (service)'), ('x_hybrid', 'hybrid X: lexicon window + IndoBERT')):
        v = base[col].astype(float)
        rows.append({'scorer': label, 'pos': int((v == 1).sum()), 'neu': int((v == 0).sum()),
                     'neg': int((v == -1).sum()), 'r_with_Y': round(v.corr(y), 3),
                     'agreement_with_M': round((v == base['x']).mean(), 3)})
    out = pd.DataFrame(rows).rename(columns={'agreement_with_M': 'agreement_with_lexicon'})
    out.insert(1, 'model_commit', commit)
    out.to_csv(os.path.join(OUTPUT, 'indobert_summary.csv'), index=False)
    print(out.drop(columns='model_commit').to_string(index=False))


if __name__ == '__main__':
    main()
