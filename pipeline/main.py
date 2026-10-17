import subprocess
import sys
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR = os.path.join(BASE_DIR, 'steps')

# Steps whose output is the record of a one-off operation are skipped when that output exists,
# so that a rerun cannot overwrite it: the scraped reviews (Step 1), the archived machine
# translation (Step 3; Google Translate output changes over time) and the top-200 stem list from
# which the service lexicon was built (Step 6). Delete the file to force the step to run.
SKIP_IF_PRESENT = {
    'step1_scraping.py':          os.path.join(BASE_DIR, 'data', 'input', 'all_reviews_raw.csv'),
    'step3_translation.py':       os.path.join(BASE_DIR, 'data', 'output', 'reviews_translated.csv'),
    'step6_extract_top_words.py': os.path.join(BASE_DIR, 'data', 'output', 'top_words.csv'),
}

steps = [
    ('Step 1 - Scraping',                'step1_scraping.py'),
    ('Step 2 - Cleaning',                'step2_cleaning.py'),
    ('Step 3 - Translation',             'step3_translation.py'),
    ('Step 4 - Dialect Normalization',   'step4_dialect_normalization.py'),
    ('Step 5 - Text Preprocessing',      'step5_text_preprocessing.py'),
    ('Step 6 - Extract Top Words',       'step6_extract_top_words.py'),
    ('Step 7 - Sentiment Scoring',       'step7_sentiment_scoring.py'),
    ('Step 8 - OLS Regression',           'step8_ols_regression.py'),
]


def run_step(name, filename):
    filepath = os.path.join(STEPS_DIR, filename)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, filepath], capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed. Pipeline stopped.")
        sys.exit(result.returncode)
    print(f"[OK] {name} done.")


if __name__ == '__main__':
    print("Starting pipeline...")

    for name, filename in steps:
        record = SKIP_IF_PRESENT.get(filename)
        if record and os.path.exists(record):
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            print(f"[SKIP] Output already exists: {record}")
            print(f"[SKIP] {name} skipped.")
            continue
        run_step(name, filename)

    print(f"\n{'='*60}")
    print("  Pipeline complete! Check results in pipeline/data/output/")
    print(f"{'='*60}\n")
