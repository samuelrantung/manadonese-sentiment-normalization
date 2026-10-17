# Processing pipeline

Collects Google Maps restaurant reviews from North Sulawesi, translates and normalizes them, and scores
each review with a lexicon-based scorer: an overall polarity *M* and a service polarity *X* (Section 2 of
the paper). The experiments in `experiments/` and `annotation/` build on its outputs; see `../README.md`
for how to run them and where each result in the paper comes from.

## Steps

| Step | Script | What it does | Output (`data/output/`) |
|---|---|---|---|
| 1 | `step1_scraping.py` | Outscraper API, queries *Restoran di Manado / Tomohon / Bitung, Sulawesi Utara*; up to 45 restaurants per query by review count, up to 30 reviews each, newest first, 2023-01-01 to 2026-02-01. **Skipped while `data/input/all_reviews_raw.csv` exists** (1,367 reviews, 57 restaurants). | `data/input/all_reviews_raw.csv` |
| 2 | `step2_cleaning.py` | Removes missing text, invalid ratings, reviews under three words and duplicates; only the length rule applied (1,367 → 1,098). | `reviews_cleaned.csv` |
| 3 | `step3_translation.py` | Google Translate (deep-translator) to Indonesian with automatic language detection; retries until every review is translated; case-folds and keeps letters only; exports the 300 most frequent words (input to the dictionary prompt). **Skipped while its output exists; the archived translation is the record.** | `reviews_translated.csv`, `data/input/top_words_raw.csv` |
| 4 | `step4_dialect_normalization.py` | Applies the 109-entry dictionary (`data/input/manado_slang_dict.json`) as case-insensitive whole-word replacements, and rewrites *salah satu* to *satu*. | `reviews_normalized.csv` |
| 5 | `step5_text_preprocessing.py` | Removes NLTK Indonesian stop-words except the negators, five evaluative words (*baik, lama, kelamaan, biasa, masalah*) and the contrast markers (*tapi, tetapi, namun, padahal, sedangkan*): 740 of 757 words removed. Stems with Sastrawi. Two reviews become empty (1,098 → 1,096). | `reviews_preprocessed.csv` |
| 6 | `step6_extract_top_words.py` | 200 most frequent stems, the input to the service-lexicon prompt. **Skipped while its output exists; the file is the record the lexicon was built from.** | `top_words.csv` |
| 7 | `step7_sentiment_scoring.py` | Lexicon scoring (below). | `reviews_scored.csv` |
| 8 | `step8_ols_regression.py` | The two OLS models of the submitted version, kept for comparison. Superseded by `experiments/robustness_checks.py`; its hypothesis verdicts are not used in the revised paper. | `ols_regression_results.txt`, `ols_coefficients.csv` |

`python main.py` runs the steps in order. Each step writes a log (`data/output/step*.log`) and a one-line
summary to `data/output/pipeline_report.txt`, which also keeps the history of earlier runs.

## Lexicon scoring (Step 7)

- **Weights.** InSet (`data/input/positive.tsv`, `negative.tsv`; −5 to +5). A word listed in both files
  gets the sum of its two weights (1,142 words do). The domain table `data/input/domain_overrides.csv`
  then overrides 135 weights that are implausible for restaurant reviews, each with a stated reason: topic
  words set to 0 (e.g. *babi*, *restoran*, *harga*), corrected signs (e.g. *bagus*, *lambat*, *mahal*) and
  missing evaluative stems (e.g. *kecewa*). The table was compiled in two rounds and frozen before the
  gold-standard evaluation.
- **Negation.** The negators (*tidak, bukan, kurang, jangan, belum, tak, tiada, tanpa, enggan*) are not
  scored; they reverse the weight of a token one or two positions after them, looked up in the whole
  review.
- **Service windows.** Around each service term (`data/input/service_keywords.txt`, 48 stems) the window
  covers five tokens on each side and stops at a contrast marker. The service region is the union of all
  windows, so each token is counted once.
- **Scores.** *M* is the sign of the summed contributions over the whole review, *X* over the service
  region (0 when no service term occurs), *M*<sub>rest</sub> over the rest of the review.

## LLM-built resources

Both resources were built with Google Gemini (web application) from frequency lists of this corpus; the
prompts are in `../llm_provenance/` as text files, next to the PDF exports of the Gemini sessions, and the dictionary prompt is also the template in
`steps/step3_translation.py`. The dictionary output was adopted as returned and then validated by two
native speakers; the service-lexicon output had two courtesy terms (*terima*, *kasih*) removed afterwards.

| Resource | Input | File | Size |
|---|---|---|---|
| Normalization dictionary | `data/input/top_words_raw.csv` (300 words) | `data/input/manado_slang_dict.json` | 109 entries |
| Service lexicon | `data/output/top_words.csv` (200 stems) | `data/input/service_keywords.txt` | 48 terms |

## Implementation notes

- **Translation errors.** The free Google Translate endpoint sometimes returns an HTML error page as an
  ordinary string. Step 3 checks every response against known error markers and retries, so error text
  never enters the corpus.
- **Punctuation.** Step 3 deletes non-letter characters rather than replacing them with a space, so words
  joined only by punctuation in the translated text (e.g. *enak,pelayanan*) become one token.
- **Paths.** All paths are resolved relative to this folder, so the code runs unmodified on any machine.
