"""
Figure 1: pipeline diagram (vector).

Layout, left to right: reviews -> translation -> normalization, then a junction that branches
into three rows. Top row: preprocessing -> lexicon scorer (outputs M, X). Middle row: hybrid
scorer (service polarity), which takes the service terms found by the lexicon scorer. Bottom
row: transformer scorer (review polarity). The transformer and hybrid branch off before
preprocessing because they read unstemmed text. Dashed boxes are the two lexical resources built
with a large language model. One accent colour marks the three scorers. Each box is sized to its
measured label, so text never overflows.

Outputs: ../../manuscript/figures/ (or data/output/figures/) figure1_pipeline.{png,pdf,svg}
Run from the pipeline root:  python experiments/figure1.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The manuscript folder when the pipeline sits in the authors' project; otherwise (for example in
# the supplementary archive) data/output/figures.
_MANUSCRIPT_FIGS = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'manuscript', 'figures'))
OUT_DIR = _MANUSCRIPT_FIGS if os.path.isdir(_MANUSCRIPT_FIGS) else os.path.join(BASE_DIR, 'data', 'output', 'figures')

INK, MUTED, EDGE = '#0b0b0b', '#52514e', '#8a8984'
FILL, ACCENT, ACCENT_FILL = '#f3f3f1', '#2a78d6', '#e8f1fc'
FONT = 8.5
PAD = 1.2          # horizontal padding inside a box, in data units
GAP = 2.6          # gap between boxes in a row
BOX_H = 7.4

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


class Canvas:
    def __init__(self, width_in=6.3, height_in=2.4, xmax=100.0, ymax=42.5):
        self.fig, self.ax = plt.subplots(figsize=(width_in, height_in))
        self.ax.set_xlim(0, xmax)
        self.ax.set_ylim(0, ymax)
        self.ax.axis('off')
        self.renderer = self.fig.canvas.get_renderer()

    def text_width(self, s, size=FONT):
        t = self.ax.text(0, 0, s, fontsize=size)
        bb = t.get_window_extent(renderer=self.renderer).transformed(self.ax.transData.inverted())
        t.remove()
        return bb.width

    def box(self, x, yc, label, kind='step', width=None):
        fc, ec, ls = {'step': (FILL, EDGE, '-'), 'scorer': (ACCENT_FILL, ACCENT, '-'),
                      'resource': ('white', MUTED, (0, (3, 2)))}[kind]
        w = width or self.text_width(label) + 2 * PAD
        y = yc - BOX_H / 2
        self.ax.add_patch(FancyBboxPatch((x, y), w, BOX_H, boxstyle='round,pad=0,rounding_size=0.9',
                                         fc=fc, ec=ec, lw=0.9, ls=ls))
        self.ax.text(x + w / 2, yc, label, ha='center', va='center', fontsize=FONT, color=INK, linespacing=1.1)
        return {'l': (x, yc), 'r': (x + w, yc), 't': (x + w / 2, y + BOX_H), 'b': (x + w / 2, y),
                'x': x, 'w': w, 'yc': yc}

    def arrow(self, a, b, color=MUTED, dashed=False):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7, lw=0.9, color=color,
                                          ls=(0, (3, 2)) if dashed else '-', shrinkA=0, shrinkB=0.6))

    def line(self, pts, color=MUTED):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=0.9, solid_capstyle='butt')


def main():
    c = Canvas()
    top, mid, bot, res = 27.5, 15.5, 4.0, 38.0

    reviews = c.box(0.3, mid, 'Reviews\n(n = 1,096)')
    trans = c.box(reviews['x'] + reviews['w'] + GAP, mid, 'Translation')
    norm = c.box(trans['x'] + trans['w'] + GAP, mid, 'Normalization')
    junction = norm['x'] + norm['w'] + GAP * 0.8
    prep = c.box(junction + GAP * 0.8, top, 'Preprocessing')

    scorer_w = max(c.text_width(s) for s in ('Lexicon\nscorer', 'Hybrid\nscorer', 'Transformer\nscorer')) + 2 * PAD
    sx = prep['x'] + prep['w'] + GAP
    lex = c.box(sx, top, 'Lexicon\nscorer', 'scorer', scorer_w)
    hyb = c.box(sx, mid, 'Hybrid\nscorer', 'scorer', scorer_w)
    trf = c.box(sx, bot, 'Transformer\nscorer', 'scorer', scorer_w)

    dic_w = c.text_width('Dictionary\n(109 entries)') + 2 * PAD
    dic = c.box(norm['x'] + norm['w'] / 2 - dic_w / 2, res, 'Dictionary\n(109 entries)', 'resource', dic_w)
    svc_w = c.text_width('Service lexicon\n(48 terms)') + 2 * PAD
    svc = c.box(sx + scorer_w / 2 - svc_w / 2, res, 'Service lexicon\n(48 terms)', 'resource', svc_w)

    c.arrow(reviews['r'], trans['l'])
    c.arrow(trans['r'], norm['l'])
    # branch from normalization: a vertical bus at the junction, then one arrow per row
    c.line([norm['r'], (junction, mid)])
    c.line([(junction, bot), (junction, top)])
    c.arrow((junction, top), prep['l'])
    c.arrow((junction, mid), hyb['l'])
    c.arrow((junction, bot), trf['l'])
    c.arrow(prep['r'], lex['l'])
    c.arrow(dic['b'], norm['t'], dashed=True)
    c.arrow(svc['b'], lex['t'], dashed=True)
    c.arrow(lex['b'], hyb['t'], color=ACCENT, dashed=True)
    c.ax.text(lex['x'] + scorer_w / 2 + 1.0, (lex['yc'] + hyb['yc']) / 2, 'service terms',
              ha='left', va='center', fontsize=7.5, color=ACCENT, style='italic')

    for b, label in ((lex, 'M, X'), (hyb, 'service\npolarity'), (trf, 'review\npolarity')):
        end = (b['x'] + b['w'] + 3.6, b['yc'])
        c.arrow(b['r'], end, color=INK)
        c.ax.text(end[0] + 0.8, end[1], label, ha='left', va='center', fontsize=FONT, color=INK, linespacing=1.1)

    right = sx + scorer_w + 3.6 + 0.8 + c.text_width('service\npolarity')
    if right > 100:
        raise SystemExit(f'layout too wide: {right:.1f} > 100')

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in (('png', {'dpi': 600}), ('pdf', {}), ('svg', {})):
        c.fig.savefig(os.path.join(OUT_DIR, f'figure1_pipeline.{ext}'), bbox_inches='tight', pad_inches=0.03, **kw)
    print(f'saved to {OUT_DIR} (rightmost text at x = {right:.1f} of 100)')


if __name__ == '__main__':
    main()
