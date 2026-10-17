# Supplementary material

**Evaluating Dialect Normalization for Aspect Sentiment in Manadonese Restaurant Reviews**
Samuel Rantung and Ronny Walean, CogITo Smart Journal (revised submission, October 2026)

This archive contains the code, data, lexical resources, LLM provenance and annotations behind every
table, figure and number in the paper. Section and table numbers below refer to the revised manuscript.

```
supplementary_material/
├── README.md                 this file
├── llm_provenance/           prompts and Gemini chat exports for the two LLM-built resources
└── pipeline/                 code, inputs, outputs, experiments and annotations
    ├── README.md             the processing pipeline, step by step
    ├── requirements.txt      pinned package versions
    ├── main.py               runs Steps 1–8
    ├── steps/                Steps 1–8 of the pipeline
    ├── experiments/          ablation, baselines, robustness, sensitivity, descriptives, figures
    ├── annotation/           gold-standard sample, annotation workbooks, agreement and evaluation
    └── data/
        ├── input/            raw reviews, lexical resources, domain table
        ├── output/           all intermediate and result files
        └── submitted_version/ outputs of the originally submitted version, for comparison
```

## Running the code

Python 3.14. From `pipeline/`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                      # Steps 2, 4, 5, 7, 8 (1, 3 and 6 are skipped, see below)
python experiments/ablation.py      # then the experiments, in the order given below
```

Every output in `pipeline/data/output/` and `pipeline/annotation/` is already included, so nothing has to
be rerun to inspect the results. Three caveats:

1. **Steps 1, 3 and 6 are records, not recomputations.** `main.py` skips them while their outputs exist.
   Step 1 needs an Outscraper API key. Step 3 calls Google Translate, whose output changes over time; the
   archived translation (`data/output/reviews_translated.csv`) is the one every result uses. Step 6
   produced the top-200 stem list (`data/output/top_words.csv`) from which the service lexicon was built.
2. **Step 8 reproduces the analysis of the submitted version** (two OLS models and their hypothesis
   verdicts). Those verdicts are not used in the revised paper, which shows that the mediation cannot be
   identified from short reviews. The rating models of the revised paper (Table 3) come from
   `experiments/robustness_checks.py`.
3. **The gold-standard sample is recorded, not redrawn.** It was drawn with a fixed seed on 14 October
   2026, when the analytical sample had 1,095 reviews; a final preprocessing fix later added one review
   (1,096). Rerunning `annotation/build_annotation_workbooks.py` on the current data therefore draws a
   different sample. `annotation/gold_sample_key.csv` is the record that links each gold review ID to its
   review, and all 300 gold reviews are in the 1,096-review corpus.

The transformer baseline downloads `w11wo/indonesian-roberta-base-sentiment-classifier` from Hugging Face
at the pinned revision `ac452dcb0f4966130bba44f4ee0013bb5d52c282` (about 500 MB on first use). The LLM
resources are not regenerated: the Gemini outputs are fixed files.

### Order of the experiments

All commands are run from `pipeline/`. Later scripts read files written by earlier ones.

| # | Command | Writes |
|---|---|---|
| 1 | `python experiments/ablation.py` | `data/output/ablation_results.csv`, `ablation_bootstrap.csv`, `ablation_predictions.csv` |
| 2 | `python experiments/indobert_baseline.py` | `data/output/indobert_predictions.csv`, `indobert_summary.csv` |
| 3 | `python annotation/check_annotations.py annotation/Anotasi_Annotator_A.xlsx annotation/Anotasi_Annotator_B.xlsx` | `annotation/merged_labels.csv` (prints agreement) |
| 4 | `python annotation/evaluate_gold.py annotation/merged_labels.csv agreed` (and `A`, `B`) | `annotation/gold_evaluation_<mode>.csv`, `gold_differences_<mode>.csv` |
| 5 | `python annotation/gold_details.py` | `annotation/gold_details.txt` |
| 6 | `python experiments/dictionary_validation.py` | `data/output/dictionary_validation.txt` |
| 7 | `python experiments/robustness_checks.py` | `data/output/robustness_checks.txt` |
| 8 | `python experiments/sensitivity.py annotation/merged_labels.csv` | `data/output/sensitivity.csv` |
| 9 | `python experiments/stopword_effect.py` | `data/output/stopword_effect.txt` |
| 9b | `python experiments/domain_table_effect.py` | `data/output/domain_table_effect.txt` |
| 10 | `python experiments/lookup_comparison.py` | `data/output/lookup_comparison.csv` (prints InSet match rates) |
| 11 | `python experiments/descriptives.py` | `data/output/descriptives.txt` |
| 12 | `python experiments/figure1.py`, `python experiments/figure2.py` | `data/output/figures/` |

## Where each result comes from

| Paper | Content | Source |
|---|---|---|
| §2.2 | Normalization dictionary (109 entries) | `pipeline/data/input/manado_slang_dict.json`; prompt `llm_provenance/prompt_dictionary.txt`; chat export `llm_provenance/NLP Original Run (…).pdf` |
| §2.3 | Service lexicon (48 terms; the 50 Gemini returned are in `data/submitted_version/`) | `pipeline/data/input/service_keywords.txt`; prompt `llm_provenance/prompt_service_lexicon.txt`; chat export `llm_provenance/Service Lexicon Run (Gemini Flash 3 High Thinking).pdf` |
| §2.4 | Lexicon scoring, Equations 1–3, Algorithm 1 | `pipeline/steps/step7_sentiment_scoring.py`; InSet `data/input/positive.tsv`, `negative.tsv`; domain table `data/input/domain_overrides.csv` (135 entries, with a reason for each) |
| §2.5 | Transformer and hybrid scorers | `experiments/indobert_baseline.py` |
| Figure 1 | Pipeline | `experiments/figure1.py` |
| §3.1 | Collection and cleaning; corpus counts by city and rating | Steps 1–2; `data/input/all_reviews_raw.csv`, `restaurant_city.csv`; `data/output/descriptives.txt` (A) |
| §3.2 | Translation share, survival of Manadonese markers, *kita*; stop-word list | Steps 3–5; `descriptives.txt` (B); `step5_text_preprocessing.py` |
| §3.3 | Gold sample, annotation, agreement (κ) | `annotation/build_annotation_workbooks.py`, `build_translation_addendum.py`, `Anotasi_Annotator_A.xlsx`, `Anotasi_Annotator_B.xlsx`, `check_annotations.py`, `merged_labels.csv`; dictionary agreement in `dictionary_validation.txt` (1) |
| §3.6 | Software versions | `pipeline/requirements.txt` |
| Table 1, §4.1 | Corpus and normalization profile | `robustness_checks.txt` (1: review length), `descriptives.txt` (B, C), `gold_details.txt` (3: Manadonese share, sampled and reweighted), `dictionary_validation.txt` (1–2: acceptance, entry types, substitutions by type), `ablation_results.csv` (reach) |
| §4.1 | Effect of applying the annotators' corrections | `dictionary_validation.txt` (3) |
| Table 2, §4.2 | Normalization conditions and scorers on the gold standard and against the rating | `gold_evaluation_agreed.csv`, `gold_differences_agreed.csv` (per annotator: `_A`, `_B`); `ablation_results.csv`, `ablation_bootstrap.csv`; `indobert_summary.csv` |
| Table 2, §4.2 | Lexicon without the domain table | `domain_table_effect.txt` |
| §4.2 | Stability of the three re-runs | `dictionary_validation.txt` (4); chat exports in `llm_provenance/` |
| §4.3 | Mention detection; whole-review transformer as service scorer; corpus-weighted accuracy (`acc_weighted`, `weighted_difference`); majority baselines; per-class results; false detections; macro-F1 | `gold_evaluation_agreed.csv`, `gold_differences_agreed.csv`, `gold_details.txt` (1, 2, 4) |
| §4.3 | Window radius and negation scope | `sensitivity.csv` |
| Figure 2, §4.4 | Polarity by star rating | `robustness_checks.txt` (7); `experiments/figure2.py` |
| Table 3, §4.4 | Rating models (OLS and ordered logit, clustered SEs), *M* model, overlap, indirect effects | `robustness_checks.txt` (0, 1, 6, 8) |

Additional analyses requested in review and reported in the response letter:
`robustness_checks.txt` (2: service-mention models; 3: effect of each scoring fix; 4: categorical and
continuous specifications; 5: ordered logit with *M*), `stopword_effect.txt` (curated vs full NLTK list),
`lookup_comparison.csv` (surface vs stem InSet lookup), `descriptives.txt` (D: service-lexicon terms and
anchor weights; E: top contributors to *M* before and after the domain table; F: multiword expressions),
and `ablation_results.csv` (reviews with and without substitutions).

## LLM provenance

`llm_provenance/` holds both prompts as text files and the PDF exports of the five Gemini sessions:

| File | Session |
|---|---|
| `Service Lexicon Run (Gemini Flash 3 High Thinking).pdf` | Service lexicon (§2.3): Gemini 3 Flash Thinking, May 2026 |
| `NLP Original Run (Gemini Flash 3 High Thinking).pdf` | Normalization dictionary (§2.2): Gemini 3 Flash Thinking, May 2026 |
| `NLP Run 1–3 (Gemini Flash 3.8 High Thinking).pdf`, `run1.json`–`run3.json` | Three re-runs of the dictionary prompt (§4.2): Gemini 3.8 Flash, October 2026 |

All sessions used the Gemini web application, which does not expose sampling settings. The re-runs sent
the prompt from the template in `steps/step3_translation.py`, in which the number of words was left as the
unfilled placeholder `{top_n}`. The output of the original sessions was adopted as returned; the dictionary
was then validated by two native speakers (§3.3), and two courtesy terms (*terima*, *kasih*) were removed
from the service lexicon.

## Data

The reviews are public Google Maps reviews of 56 restaurants in Manado, Tomohon and Bitung (1 January
2023 to 31 January 2026). Reviewer names were replaced by a salted SHA-256 hash truncated to 12 characters
before any analysis; the salt is not distributed. Review text, ratings, place names and dates are
reproduced as published. The data are shared for the reproduction of this study only; please do not
attempt to identify reviewers. The two annotators are identified only as A and B. The annotation
workbooks are in Indonesian (sheets *Panduan* = guideline, *Ulasan* = reviews, *Ulasan_terjemahan* =
translated non-Indonesian reviews, *Kamus* = dictionary validation).

Third-party resources included for reproducibility: the InSet lexicon (Koto and Rahmaningtyas, 2017;
`positive.tsv`, `negative.tsv`) and the Colloquial Indonesian Lexicon (Salsabila et al., 2018;
`colloquial-indonesian-lexicon.csv`). Please cite the original publications when reusing them.

The analysis code was written with the assistance of an AI model (Claude, Anthropic) and verified by the
authors; see the AI-use statement in Section 6 of the paper.
