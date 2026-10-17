"""
Builds the translation addendum for the gold-set annotation.

Non-Indonesian reviews in the gold sample are skipped in the main workbooks and labelled
here instead, from the pipeline's own Indonesian translation (the text the pipeline scored),
so that both annotators judge the same text rather than translating independently.
Review IDs are unchanged, so labels merge back into the main sample by ID.

Languages were identified automatically (script detection and English stop-word share)
and then corrected by hand; the hand-checked assignment is the LANGUAGE table below.

Output: annotation/Anotasi_Tambahan_Terjemahan_A.xlsx and _B.xlsx (identical)
Run from the pipeline root:  python annotation/build_translation_addendum.py
"""
import os
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'annotation')
sys.path.insert(0, OUT_DIR)
from build_annotation_workbooks import dropdown, header  # noqa: E402

LANGUAGE = {
    **{i: 'Korea' for i in ['R028', 'R032', 'R137']},
    **{i: 'Jerman' for i in ['R023', 'R025', 'R104', 'R138', 'R142', 'R171']},
    **{i: 'Prancis' for i in ['R085', 'R144', 'R156', 'R259', 'R260']},
    **{i: 'Belanda' for i in ['R089', 'R098', 'R134', 'R152', 'R161', 'R164']},
    'R300': 'Italia',
    **{i: 'Inggris' for i in [
        'R018', 'R022', 'R027', 'R031', 'R038', 'R039', 'R052', 'R055', 'R059', 'R062', 'R078',
        'R092', 'R094', 'R097', 'R106', 'R110', 'R114', 'R115', 'R118', 'R122', 'R125', 'R129',
        'R130', 'R145', 'R151', 'R153', 'R158', 'R160', 'R167', 'R176', 'R177', 'R183', 'R188',
        'R193', 'R194', 'R197', 'R199', 'R201', 'R214', 'R233', 'R238', 'R239', 'R254', 'R264',
        'R278', 'R280', 'R289', 'R290', 'R291', 'R296', 'R298']},
}

FONT, BOLD = Font(name='Arial', size=10), Font(name='Arial', size=10, bold=True)
HEAD = Font(name='Arial', size=10, bold=True, color='FFFFFF')
HEADFILL = PatternFill('solid', fgColor='305496')
THIN = Border(*(Side(style='thin', color='BFBFBF'),) * 4)
WRAP = Alignment(wrap_text=True, vertical='top')


GREY = Font(name='Arial', size=9, color='808080')
INPUT = PatternFill('solid', fgColor='FFF2CC')


def main():
    key = pd.read_csv(os.path.join(OUT_DIR, 'gold_sample_key.csv')).set_index('review_id')
    rows = sorted(LANGUAGE.items())
    n = len(rows)

    for tag in ('A', 'B'):
        wb = Workbook()
        ws = wb.active
        ws.title = 'Ulasan_terjemahan'
        header(ws, [('ID', 7), ('Bahasa asli', 10), ('Ulasan (terjemahan Indonesia)', 60),
                    ('Bahas pelayanan?', 13), ('Sentimen pelayanan', 16), ('Sentimen keseluruhan', 16),
                    ('Ada bahasa Manado?', 13), ('Catatan (opsional)', 24), ('Ulasan asli (referensi)', 45)])
        for i, (rid, lang) in enumerate(rows, start=2):
            vals = [rid, lang, key.loc[rid, 'translated_text'], None, None, None, 'Tidak', None,
                    key.loc[rid, 'original_text']]
            for c, v in enumerate(vals, start=1):
                cell = ws.cell(row=i, column=c, value=v)
                cell.font, cell.border, cell.alignment = (GREY if c == 9 else FONT), THIN, WRAP
                if 4 <= c <= 8:
                    cell.fill = INPUT
        dropdown(ws, 'D', ['Ya', 'Tidak'], n)
        dropdown(ws, 'E', ['Positif', 'Negatif', 'Netral-campuran'], n)
        dropdown(ws, 'F', ['Positif', 'Negatif', 'Netral-campuran'], n)
        dropdown(ws, 'G', ['Ya', 'Tidak'], n)

        g = wb.create_sheet('Panduan', 0)
        g.column_dimensions['A'].width = 110
        notes = [
            ('Lembar tambahan: ulasan bahasa asing', Font(name='Arial', size=14, bold=True)),
            ('Ulasan-ulasan ini DILEWATI di file utama dan dinilai di sini.', BOLD),
            ('Nilai berdasarkan kolom "Ulasan (terjemahan Indonesia)", yaitu terjemahan yang dipakai sistem '
             '(huruf kecil, tanpa tanda baca). Jangan menerjemahkan sendiri.', FONT),
            ('Cara mengisi sama seperti file utama (lihat sheet Panduan di file utama). Isi hanya sel kuning.', FONT),
            ('"Ada bahasa Manado?" sudah diisi Tidak karena semua ulasan di sini berbahasa asing.', FONT),
            ('Ulasan bahasa Inggris yang SUDAH dinilai di file utama boleh dikosongkan di sini.', FONT),
            ('Kolom "Ulasan asli" hanya untuk referensi.', GREY),
            ('Jumlah ulasan: %d (Inggris 51, Jerman 6, Belanda 6, Prancis 5, Korea 3, Italia 1).' % n, FONT),
        ]
        for r, (text, font) in enumerate(notes, start=1):
            g.cell(row=r, column=1, value=text).font = font
        wb.active = 1
        wb.save(os.path.join(OUT_DIR, f'Anotasi_Tambahan_Terjemahan_{tag}.xlsx'))

    print(f'{n} reviews per workbook; languages: {pd.Series(LANGUAGE).value_counts().to_dict()}')


if __name__ == '__main__':
    main()
