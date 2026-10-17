import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import log_step


# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE            = os.path.join(BASE_DIR, 'data', 'output', 'reviews_preprocessed.csv')
OUTPUT_FILE           = os.path.join(BASE_DIR, 'data', 'output', 'reviews_scored.csv')
POS_LEXICON_FILE      = os.path.join(BASE_DIR, 'data', 'input',  'positive.tsv')
NEG_LEXICON_FILE      = os.path.join(BASE_DIR, 'data', 'input',  'negative.tsv')
SERVICE_KEYWORDS_FILE = os.path.join(BASE_DIR, 'data', 'input',  'service_keywords.txt')
# Optional domain corrections to InSet (word, new_weight, reason). Applied only if present.
DOMAIN_OVERRIDES_FILE = os.path.join(BASE_DIR, 'data', 'input',  'domain_overrides.csv')

# Negation words that flip the sentiment of the word that follows them.
# These are preserved from stopword removal in step5.
NEGATION_WORDS = {'tidak', 'bukan', 'kurang', 'jangan', 'belum', 'tak', 'tiada', 'tanpa', 'enggan'}

# Clause-boundary markers; service windows stop here (kept from stopword removal in step5).
CONTRAST_WORDS = {'tapi', 'tetapi', 'namun', 'padahal', 'sedangkan'}


# ==========================================
# 2. LOADERS
# ==========================================
_GEMINI_PROMPT = """
==============================================================
  COPY-PASTE PROMPT FOR GEMINI AI
==============================================================

Peran: Kamu adalah pakar linguistik dan asisten penelitian untuk Aspect-Based Sentiment Analysis (ABSA) pada ulasan restoran platform Google Maps di Sulawesi Utara.
Konteks Data:
1. Saya sedang mengekstraksi Variabel X yaitu "Kualitas Layanan / Employee Behaviour" (Fokus HANYA pada interaksi manusia, staf, dan kecepatan layanan).
2. Teks ulasan berasal dari bahasa formal, informal, serta dialek lokal Manado yang sudah dinormalisasi.
3. Teks ulasan SUDAH MELALUI PROSES STEMMING (Sastrawi). Artinya, semua kata di dalam data saya murni berupa KATA DASAR tanpa awalan/akhiran.
TUGASMU: Saya akan mengunggah file bernama top_words.csv yang berisi daftar 200 kata dengan frekuensi kemunculan tertinggi dari data penelitian saya.
Tugasmu adalah menyeleksi dan mengekstraksi kata-kata dari file tersebut yang relevan dengan aspek "Kualitas Layanan / Employee Behaviour", lalu kembangkan menjadi daftar kamus kata kunci (Service Quality Lexicon).
Kriteria Pemilihan & Pengembangan Kata:
1. Mencakup aktor/subjek layanan (contoh: kasir, staf, koki, pelayan).
2. Mencakup atribut kinerja karyawan (contoh: ramah, cepat, lambat, lelet, senyum).
3. Mencakup kata dialek Manado terkait layanan (yang sudah dalam bentuk dasar/normalisasi).
SANGAT PENTING (BATASAN EXCLUSION):
1. WAJIB KATA DASAR: Semua kata yang kamu pilih dan kembangkan WAJIB dalam bentuk KATA DASAR (contoh: tulis "layan", jangan "pelayanan").
2. DILARANG memasukkan kata sentimen global atau evaluasi umum untuk mencegah tumpang tindih variabel kepuasan (seperti: mantap, puas, senang, rekomendasi, buruk, kecewa).
3. DILARANG memasukkan atribut fisik restoran atau makanan (seperti: nyaman, bersih, kotor, enak, jorok).
FORMAT OUTPUT: 
- Tanpa komentar, tanpa penomoran, tanpa duplikasi dari seed words
- Siap copy-paste langsung ke file .txt

Jika kamu sudah mengerti instruksi ini, jawab dengan "Saya mengerti. Silakan unggah file top_words.csv Anda."

==============================================================
"""


def load_service_keywords(filepath):
    """
    Loads the service quality keyword list from a plain text file (one word per line).
    This file is produced bottom-up from step6 top words, then expanded by Gemini AI.
    Returns a lowercase set for O(1) membership testing.

    If the file does not exist, logs a warning with a copy-paste ready Gemini AI prompt
    to help the user generate the file, then exits the pipeline.
    """
    if not os.path.exists(filepath):
        logging.warning("=" * 62)
        logging.warning("  SERVICE KEYWORDS FILE NOT FOUND")
        logging.warning("=" * 62)
        logging.warning(f"  Expected: {filepath}")
        logging.warning("")
        logging.warning("  Step 7 cannot run without this file.")
        logging.warning("  To generate it:")
        logging.warning("    1. Run Step 6 to produce top_words.csv")
        logging.warning("    2. Feed top_words.csv to Gemini AI using the prompt below")
        logging.warning("    3. Save Gemini's output as: service_keywords.txt")
        logging.warning("    4. Place the file in: pipeline/data/input/")
        logging.warning("")
        for line in _GEMINI_PROMPT.strip().splitlines():
            logging.warning(f"  {line}")
        logging.warning("=" * 62)
        raise SystemExit(1)

    with open(filepath, encoding='utf-8') as f:
        keywords = {line.strip().lower() for line in f if line.strip()}
    logging.info(f"Service keywords loaded: {len(keywords)} words from {filepath}")
    return keywords


def load_lexicon(pos_file, neg_file):
    """
    Loads the InSet sentiment lexicon from two TSV files (positive and negative).

    InSet scores each word independently on a positive and a negative dimension, so a
    word can appear in both files. The polarity used here is the sum of the two weights.
    Returns a dict mapping word -> net weight (> 0 positive, < 0 negative, 0 neutral).
    """
    if not (os.path.exists(pos_file) and os.path.exists(neg_file)):
        raise FileNotFoundError(
            f"InSet lexicon files are required.\n"
            f"  Expected: {pos_file}\n"
            f"            {neg_file}"
        )

    pos_df = pd.read_csv(pos_file, sep='\t')
    neg_df = pd.read_csv(neg_file, sep='\t')

    # InSet annotates each word on a positive AND a negative dimension, so a word may
    # appear in both files with different weights (1,142 words do). Its polarity is the
    # SUM of the two. Weights are therefore accumulated, not assigned: a plain assignment
    # would let the second file silently overwrite the first, inverting the polarity of
    # the corpus's most frequent evaluative words (e.g. 'enak' +5/-4 would become -4
    # instead of its true net +1).
    lexicon = {}
    for _, row in pos_df.iterrows():
        word = str(row['word']).strip().lower()
        lexicon[word] = lexicon.get(word, 0.0) + float(row['weight'])
    for _, row in neg_df.iterrows():
        word = str(row['word']).strip().lower()
        lexicon[word] = lexicon.get(word, 0.0) + float(row['weight'])

    overlap = len(set(pos_df['word'].astype(str).str.strip().str.lower())
                  & set(neg_df['word'].astype(str).str.strip().str.lower()))

    logging.info(
        f"Lexicon loaded: {len(pos_df)} positive words, {len(neg_df)} negative words, "
        f"{len(lexicon)} unique terms ({overlap} appear in both files and were summed)."
    )

    # Domain corrections: InSet assigns polarity to many neutral restaurant nouns
    # (e.g. babi -4, restoran -4, harga +3) and inverts some evaluative terms
    # (bagus -2, lambat +1). Overrides replace the net weight; the file documents each reason.
    if os.path.exists(DOMAIN_OVERRIDES_FILE):
        ov = pd.read_csv(DOMAIN_OVERRIDES_FILE)
        for word, weight in zip(ov['word'].astype(str).str.strip().str.lower(), ov['new_weight']):
            lexicon[word] = float(weight)
        logging.info(f"Domain overrides applied: {len(ov)} words from {DOMAIN_OVERRIDES_FILE}")
    else:
        logging.info("No domain_overrides.csv found; using raw InSet weights.")
    return lexicon


# ==========================================
# 3. SCORING FUNCTIONS
# ==========================================
def _token_score(words, i, lexicon):
    """
    Returns the signed contribution of the token at position i.

    Negation words are operators, not evaluations: they are never scored themselves
    (InSet weights 'tidak' at -5, which would otherwise add negative mass to every
    negated sentence). A scored token is flipped (x -1) if a negation word occurs
    1 or 2 positions before it, looked up in the FULL token sequence so that a
    negator just outside a service window is still observed.
    """
    word = words[i].lower()
    if word in NEGATION_WORDS:
        return 0.0
    weight = lexicon.get(word, 0.0)
    if weight == 0.0:
        return 0.0
    negated = any(words[i - k].lower() in NEGATION_WORDS for k in (1, 2) if i - k >= 0)
    return -weight if negated else weight


def calculate_sentiment_score(text, lexicon):
    """
    Computes a raw document-level sentiment score for a review using InSet lexicon weights,
    summing _token_score over every token. Returns a float: positive > 0, negative < 0, neutral = 0.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = text.split()
    return round(sum(_token_score(words, i, lexicon) for i in range(len(words))), 4)


def classify_sentiment(score):
    """
    Converts a raw sentiment score to a categorical label.
    Returns 1 (Positive), -1 (Negative), or 0 (Neutral).
    """
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0


ABSA_WINDOW_SIZE = 5  # Number of words to scan on each side of a matched keyword


def service_window_positions(words, keyword_set, window=ABSA_WINDOW_SIZE, stop_at_contrast=True):
    """
    Returns the sorted UNION of token positions within +/-`window` of any service keyword.
    Using the union means a token near several keywords (e.g. 'layan ramah cepat')
    is counted once, not once per overlapping window.

    With stop_at_contrast, a window does not extend past a contrast marker, so that in
    'makan enak tapi layan lama' the food evaluation stays outside the service window.
    """
    positions = set()
    for i, w in enumerate(words):
        if w.lower() not in keyword_set:
            continue
        positions.add(i)
        for step in (-1, 1):
            j = i + step
            while 0 <= j < len(words) and abs(j - i) <= window:
                if stop_at_contrast and words[j].lower() in CONTRAST_WORDS:
                    break
                positions.add(j)
                j += step
    return sorted(positions)


def score_absa_service_quality(text, keyword_set, lexicon, window=ABSA_WINDOW_SIZE):
    """
    Performs Aspect-Based Sentiment Analysis (ABSA) for the service quality aspect:
    sums _token_score over the union of service windows and returns its sign.

    Returns:
       1  — service aspect is present and sentiment is positive
      -1  — service aspect is present and sentiment is negative
       0  — no service keyword found (aspect absent), or sentiment is neutral
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    words = text.split()
    total = sum(_token_score(words, j, lexicon)
                for j in service_window_positions(words, keyword_set, window))
    return classify_sentiment(total)


# ==========================================
# 4. MAIN PIPELINE STEP
# ==========================================
def run_sentiment_scoring(input_file, output_file, pos_lexicon_file, neg_lexicon_file, service_keywords_file):
    """
    Scores each review for customer satisfaction (M) and service quality (X).

    Columns added:
      - satisfaction_raw_M   : raw float score (sum of InSet lexicon weights)
      - customer_satisfaction_M : categorical label: 1 (Positive), -1 (Negative), 0 (Neutral)
      - service_quality_X    : ABSA score: 1 (positive service), -1 (negative service), 0 (absent)

    These columns feed directly into the OLS regression model:
      Y (review_rating) ~ X (service_quality) + M (customer_satisfaction)
    """
    # ---- Load all inputs first (fail fast before any computation) ----
    logging.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    logging.info(f"Loaded {total_rows} rows.")

    # Validate required columns
    required_columns = {'review_text', 'review_rating'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df['review_text'] = df['review_text'].fillna('')

    # Load all resources before computation begins
    lexicon = load_lexicon(pos_lexicon_file, neg_lexicon_file)
    keyword_set = load_service_keywords(service_keywords_file)

    # ---- Compute Variable M (Customer Satisfaction) ----
    logging.info("Computing customer satisfaction scores (Variable M)...")
    df['satisfaction_raw_M'] = df['review_text'].apply(
        lambda text: calculate_sentiment_score(text, lexicon)
    )
    df['customer_satisfaction_M'] = df['satisfaction_raw_M'].apply(classify_sentiment)

    counts = df['customer_satisfaction_M'].value_counts()
    pos_count = counts.get(1, 0)
    neg_count = counts.get(-1, 0)
    neu_count = counts.get(0, 0)
    logging.info(
        f"Customer satisfaction (M) distribution — Positive: {pos_count}, "
        f"Negative: {neg_count}, Neutral: {neu_count}"
    )

    # ---- Compute Variable X (ABSA Service Quality) ----
    logging.info(f"Computing ABSA service quality scores (Variable X, window=±{ABSA_WINDOW_SIZE})...")
    df['service_quality_X'] = df['review_text'].apply(
        lambda text: score_absa_service_quality(text, keyword_set, lexicon)
    )

    x_counts = df['service_quality_X'].value_counts()
    x_pos = x_counts.get(1, 0)
    x_neg = x_counts.get(-1, 0)
    x_abs = x_counts.get(0, 0)
    logging.info(
        f"Service quality (X) distribution — "
        f"Positive: {x_pos}, Negative: {x_neg}, Absent: {x_abs}"
    )

    # ---- Save lean output (only columns needed for OLS regression) ----
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out = pd.DataFrame({
        'google_id':            df['google_id'],      # restaurant id, for cluster-robust SEs
        'review_author':        df['review_author'],  # hashed reviewer id, for joining annotations
        'review_rating':        df['review_rating'],
        'satisfaction_raw_M':   df['satisfaction_raw_M'],
        'customer_satisfaction_M': df['customer_satisfaction_M'],
        'service_quality_X':    df['service_quality_X'],
    })
    df_out.to_csv(output_file, index=False)
    logging.info(f"Output saved to: {output_file}")

    log_step(
        'Step 7 - Sentiment Scoring',
        f"{total_rows:,} rows scored | "
        f"M (customer satisfaction): {pos_count} pos / {neg_count} neg / {neu_count} neu | "
        f"X (ABSA): {x_pos} pos / {x_neg} neg / {x_abs} absent"
    )


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'data', 'output'), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(
                os.path.join(BASE_DIR, 'data', 'output', 'step7_sentiment_scoring.log'),
                encoding='utf-8'
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_sentiment_scoring(INPUT_FILE, OUTPUT_FILE, POS_LEXICON_FILE, NEG_LEXICON_FILE, SERVICE_KEYWORDS_FILE)
