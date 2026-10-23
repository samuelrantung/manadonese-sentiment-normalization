"""
Robustness checks for the S-O-R regressions requested in review.

Uses the main pipeline output (reviews_preprocessed.csv) and the Step 7 scoring functions.
  1. Decoupling: M_rest = satisfaction computed only OUTSIDE the service windows
  2. Service mention: separate 'no service term' from 'balanced service score'
  3. Trace: the direct X -> Y coefficient after each scoring fix, from the submitted version on
  4. Functional form: categorical X/M dummies, and raw (continuous) scores
  5. Ordered logit for the 1-5 star rating
  6. Cluster-bootstrap confidence interval for the indirect effect a x b
  7. Cross-tabulations of M and X by star rating
  8. Main RQ3 models: Y ~ X + M_rest + S (and the comparison with M)
  9. Main RQ3 model with service sentiment as categories (none / positive / negative / balanced)
Every regression uses standard errors clustered by restaurant (google_id).

Report: data/output/robustness_checks.txt
Run from the pipeline root:  python experiments/robustness_checks.py
"""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "steps"))

from step7_sentiment_scoring import (  # noqa: E402
    NEGATION_WORDS,
    NEG_LEXICON_FILE,
    POS_LEXICON_FILE,
    SERVICE_KEYWORDS_FILE,
    _token_score,
    classify_sentiment,
    load_lexicon,
    load_service_keywords,
    service_window_positions,
)

OUTPUT = os.path.join(BASE_DIR, "data", "output")
REPORT = os.path.join(OUTPUT, "robustness_checks.txt")


def cluster_ols(formula, d):
    return smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["google_id"]}
    )


def coef_line(fit, term, label=None):
    return (
        f"  {label or term:28} b = {fit.params[term]:7.3f}   SE = {fit.bse[term]:.3f}   "
        f"p = {fit.pvalues[term]:.4f}"
    )


# ---------- scores ----------
def score_review(text, lexicon, keywords):
    words = text.split()
    window = set(service_window_positions(words, keywords))
    contrib = [_token_score(words, i, lexicon) for i in range(len(words))]
    m_raw = sum(contrib)
    x_raw = sum(contrib[i] for i in window)
    rest_raw = m_raw - x_raw
    scored_tokens = [i for i, c in enumerate(contrib) if c != 0]
    return {
        "x_raw": x_raw,
        "m_raw": m_raw,
        "m_rest_raw": rest_raw,
        "mention": int(bool(window)),
        "n_tokens": len(words),
        "n_scored": len(scored_tokens),
        "n_scored_in_window": sum(1 for i in scored_tokens if i in window),
    }


def legacy_scores(text, lexicon, keywords, skip_neg, full_lookback, union):
    """Scoring as it was at each stage of the fixes (stage 0 = submitted version)."""
    words = text.split()

    def tw(j, lo):
        w = words[j]
        if skip_neg and w in NEGATION_WORDS:
            return 0.0
        wt = lexicon.get(w, 0.0)
        if wt == 0.0:
            return 0.0
        start = 0 if full_lookback else lo
        neg = any(words[j - k] in NEGATION_WORDS for k in (1, 2) if j - k >= start)
        return -wt if neg else wt

    m = sum(tw(j, 0) for j in range(len(words)))
    hits = [i for i, w in enumerate(words) if w in keywords]
    if union:
        pos = sorted(
            {j for i in hits for j in range(max(0, i - 5), min(len(words), i + 6))}
        )
        x = sum(tw(j, 0) for j in pos)
    else:
        x = 0.0
        for i in hits:
            lo = max(0, i - 5)
            x += sum(tw(j, lo) for j in range(lo, min(len(words), i + 6)))
    return classify_sentiment(m), classify_sentiment(x)


def main():
    df = pd.read_csv(os.path.join(OUTPUT, "reviews_preprocessed.csv"))
    df["review_text"] = df["review_text"].fillna("")
    with contextlib.redirect_stderr(io.StringIO()):
        lexicon = load_lexicon(POS_LEXICON_FILE, NEG_LEXICON_FILE)
        keywords = load_service_keywords(SERVICE_KEYWORDS_FILE)

    s = pd.DataFrame([score_review(t, lexicon, keywords) for t in df["review_text"]])
    d = pd.concat(
        [df[["google_id", "review_rating"]].reset_index(drop=True), s], axis=1
    )
    d["y"] = d["review_rating"].astype(float)
    d["x"] = d["x_raw"].map(classify_sentiment)
    d["m"] = d["m_raw"].map(classify_sentiment)
    d["m_rest"] = d["m_rest_raw"].map(classify_sentiment)

    out = []
    p = out.append
    p(
        "ROBUSTNESS CHECKS  (n = %d reviews, %d restaurants; SEs clustered by restaurant)\n"
        % (len(d), d.google_id.nunique())
    )

    # ---- 0. main model with clustered SEs ----
    p("0. MAIN MODELS (current scoring, clustered SEs)")
    a_fit = cluster_ols("m ~ x", d)
    m2 = cluster_ols("y ~ x + m", d)
    p(coef_line(a_fit, "x", "Model 1  a: X -> M"))
    p(coef_line(m2, "x", "Model 2  c': X -> Y (direct)"))
    p(coef_line(m2, "m", "Model 2  b: M -> Y"))
    p(f"  R2 Model 1 = {a_fit.rsquared:.3f}, Model 2 = {m2.rsquared:.3f}")
    p("  Total effect (Y ~ X alone):")
    p(coef_line(cluster_ols("y ~ x", d), "x", "c: X -> Y (total)"))

    # ---- 1. decoupling ----
    p("\n1. DECOUPLING: M_rest = satisfaction from text OUTSIDE the service windows")
    sc = d[d.n_scored > 0]
    share_in_window = sc.n_scored_in_window.sum() / sc.n_scored.sum()
    xnz = d[d.x != 0]
    p(
        f"  Median review length after preprocessing: {d.n_tokens.median():.0f} tokens "
        f"(IQR {d.n_tokens.quantile(.25):.0f}-{d.n_tokens.quantile(.75):.0f})"
    )
    p(
        f"  Share of M's scored tokens that lie inside service windows: {share_in_window:.1%}"
    )
    p(
        f"  Reviews with X != 0: {len(xnz)}; of these, X == M in {(xnz.x == xnz.m).mean():.1%}; "
        f"window covers the whole review in {(xnz.n_scored_in_window == xnz.n_scored).mean():.1%}"
    )
    p(
        f"  r(X, M) = {d.x.corr(d.m):.3f}   r(X, M_rest) = {d.x.corr(d.m_rest):.3f}   "
        f"r(M_rest, Y) = {d.m_rest.corr(d.y):.3f}"
    )
    a2 = cluster_ols("m_rest ~ x", d)
    m2r = cluster_ols("y ~ x + m_rest", d)
    p(coef_line(a2, "x", "a: X -> M_rest"))
    p(coef_line(m2r, "x", "c': X -> Y (direct)"))
    p(coef_line(m2r, "m_rest", "b: M_rest -> Y"))
    p(f"  R2 = {m2r.rsquared:.3f}")

    # ---- 2. service mention ----
    p('\n2. SERVICE MENTION (X = 0 mixes "no service term" with "balanced")')
    p(
        f"  Reviews mentioning service: {d.mention.sum()} ({d.mention.mean():.1%}); "
        f"mentioned but X = 0: {((d.mention == 1) & (d.x == 0)).sum()}"
    )
    mm = cluster_ols("y ~ x + m + mention", d)
    p("  (a) all reviews, with a mention indicator:")
    for t in ("x", "m", "mention"):
        p("  " + coef_line(mm, t))
    sub = d[d.mention == 1]
    p(f"  (b) only reviews that mention service (n = {len(sub)}):")
    f1 = cluster_ols("y ~ x + m", sub)
    p("  " + coef_line(f1, "x", "c': X -> Y, with M"))
    p("  " + coef_line(f1, "m", "b: M -> Y"))
    f2 = cluster_ols("y ~ x + m_rest", sub)
    p("  " + coef_line(f2, "x", "c': X -> Y, with M_rest"))
    p("  " + coef_line(f2, "m_rest", "b: M_rest -> Y"))

    # ---- 3. trace of the direct effect across the fixes ----
    p(
        "\n3. TRACE: direct effect of X after each scoring change (Y ~ X + M; all stages on the current preprocessing)"
    )
    raw_lex = {}
    with contextlib.redirect_stderr(io.StringIO()):
        import step7_sentiment_scoring as s7

        saved = s7.DOMAIN_OVERRIDES_FILE
        s7.DOMAIN_OVERRIDES_FILE = os.path.join(OUTPUT, "__none__")
        raw_lex = s7.load_lexicon(POS_LEXICON_FILE, NEG_LEXICON_FILE)
        s7.DOMAIN_OVERRIDES_FILE = saved
    old_keys = keywords | {"terima", "kasih"}
    stages = [
        ("0 original scoring rules", raw_lex, old_keys, False, False, False),
        ("1 negators not scored", raw_lex, old_keys, True, False, False),
        ("2 full-review negation lookback", raw_lex, old_keys, True, True, False),
        ("3 merged windows", raw_lex, old_keys, True, True, True),
        ("4 drop terima/kasih", raw_lex, keywords, True, True, True),
        ("5 domain overrides (current)", lexicon, keywords, True, True, True),
    ]
    p(f"  {'stage':34} {'r(M,Y)':>7} {'b_X':>7} {'p_X':>7} {'b_M':>7} {'R2':>6}")
    for name, lx, keys, sn, fl, un in stages:
        res = [legacy_scores(t, lx, keys, sn, fl, un) for t in df["review_text"]]
        t = d[["google_id", "y"]].copy()
        t["m"], t["x"] = zip(*res)
        f = cluster_ols("y ~ x + m", t)
        p(
            f"  {name:34} {t.m.corr(t.y):7.3f} {f.params['x']:7.3f} {f.pvalues['x']:7.4f} "
            f"{f.params['m']:7.3f} {f.rsquared:6.3f}"
        )

    # ---- 4. functional form ----
    p("\n4. FUNCTIONAL FORM")
    cat = cluster_ols("y ~ C(x, Treatment(0)) + C(m, Treatment(0))", d)
    p("  (a) X and M as categories (reference = 0):")
    for t in cat.params.index[1:]:
        p(
            "  "
            + coef_line(
                cat,
                t,
                t.replace("Treatment(0)", "").replace("C(", "").replace(", )", ""),
            )
        )
    for v in ("x_raw", "m_raw", "m_rest_raw"):
        d[v + "_z"] = (d[v] - d[v].mean()) / d[v].std()
    raw = cluster_ols("y ~ x_raw_z + m_raw_z", d)
    p("  (b) raw scores, standardized (1 SD change):")
    p("  " + coef_line(raw, "x_raw_z"))
    p("  " + coef_line(raw, "m_raw_z"))
    p(f"      R2 = {raw.rsquared:.3f}")
    raw2 = cluster_ols("y ~ x_raw_z + m_rest_raw_z", d)
    p("  " + coef_line(raw2, "x_raw_z", "x_raw_z (with M_rest raw)"))
    p("  " + coef_line(raw2, "m_rest_raw_z"))

    # ---- 5. ordered logit ----
    p("\n5. ORDERED LOGIT for the 1-5 star rating (clustered SEs)")
    for label, cols in (("with M", ["x", "m"]), ("with M_rest", ["x", "m_rest"])):
        om = OrderedModel(d["review_rating"].astype(int), d[cols], distr="logit")
        fit = om.fit(
            method="bfgs",
            disp=0,
            maxiter=2000,
            cov_type="cluster",
            cov_kwds={"groups": pd.factorize(d["google_id"])[0]},
        )
        p(f"  {label}:")
        for c in cols:
            p(
                f"    {c:10} log-odds = {fit.params[c]:6.3f}  (odds ratio {np.exp(fit.params[c]):5.2f})  "
                f"SE = {fit.bse[c]:.3f}  p = {fit.pvalues[c]:.4f}"
            )

    # ---- 6. bootstrap indirect effect ----
    p(
        "\n6. INDIRECT EFFECT a x b: cluster bootstrap over restaurants (2,000 draws, percentile 95% CI)"
    )
    rng = np.random.default_rng(42)
    ids = d.google_id.unique()
    groups = {g: np.flatnonzero(d.google_id.values == g) for g in ids}
    X, Y = d.x.values.astype(float), d.y.values
    for mname in ("m", "m_rest"):
        M = d[mname].values.astype(float)

        def ab(idx):
            x, m, y = X[idx], M[idx], Y[idx]
            a = np.polyfit(x, m, 1)[0]
            A = np.column_stack([np.ones_like(x), x, m])
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
            return a * coef[2], coef[1]

        est, direct = ab(np.arange(len(d)))
        draws = np.array(
            [
                ab(np.concatenate([groups[g] for g in rng.choice(ids, len(ids))]))
                for _ in range(2000)
            ]
        )
        lo, hi = np.percentile(draws[:, 0], [2.5, 97.5])
        dlo, dhi = np.percentile(draws[:, 1], [2.5, 97.5])
        p(
            f"  mediator {mname:7}: indirect = {est:.3f} [{lo:.3f}, {hi:.3f}]   "
            f"direct = {direct:.3f} [{dlo:.3f}, {dhi:.3f}]   "
            f"share mediated = {est / (est + direct):.1%}"
        )

    # ---- 7. cross-tabulations ----
    p("\n7. CROSS-TABULATIONS (row % within each star rating)")
    for v in ("m", "x"):
        ct = pd.crosstab(d.review_rating, d[v], normalize="index").mul(100).round(1)
        ct.columns = [{-1: "neg", 0: "zero", 1: "pos"}[c] for c in ct.columns]
        ct["n"] = d.review_rating.value_counts().sort_index()
        p(
            f"  {v.upper()} by star rating:\n"
            + "\n".join("    " + l for l in ct.to_string().splitlines())
        )

    # ---- 8. main RQ3 models as specified in Section 3.5 ----
    p(
        "\n8. MAIN RQ3 MODELS: Y ~ X + M_rest + S (and the comparison with M), clustered SEs"
    )
    groups = pd.factorize(d["google_id"])[0]
    for label, cols in (
        ("with M_rest", ["x", "m_rest", "mention"]),
        ("with M", ["x", "m", "mention"]),
    ):
        ols = cluster_ols("y ~ " + " + ".join(cols), d)
        om = OrderedModel(d["review_rating"].astype(int), d[cols], distr="logit").fit(
            method="bfgs",
            disp=0,
            maxiter=2000,
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
        p(f"  {label}:   OLS R2 = {ols.rsquared:.3f}")
        for c in cols:
            p(
                f"    {c:8} OLS b = {ols.params[c]:6.3f} (SE {ols.bse[c]:.3f}, p = {ols.pvalues[c]:.4f})   "
                f"ordered logit = {om.params[c]:6.3f} (SE {om.bse[c]:.3f}, p = {om.pvalues[c]:.4f}, "
                f"OR {np.exp(om.params[c]):.2f})"
            )

    # ---- 9. main RQ3 model with service sentiment as categories (B-D34) ----
    p(
        "\n9. MAIN RQ3 MODEL, SERVICE SENTIMENT AS CATEGORIES: Y ~ service + M_rest (reference = no service mention)"
    )
    d["service"] = np.where(
        d["mention"] == 0,
        "none",
        d["x"].map({1: "positive", -1: "negative", 0: "balanced"}),
    )
    p("  " + ", ".join(f"{k} {v}" for k, v in d["service"].value_counts().items()))
    svc = cluster_ols("y ~ C(service, Treatment('none')) + m_rest", d)
    for t in svc.params.index[1:]:
        p(
            "  "
            + coef_line(svc, t, t.replace("C(service, Treatment('none'))", "service"))
        )
    p(f"      R2 = {svc.rsquared:.3f}")
    lin = cluster_ols("y ~ x + m_rest + mention", d)
    p(
        f"  Implied by the linear model in section 8: positive {lin.params['mention'] + lin.params['x']:+.3f}, "
        f"negative {lin.params['mention'] - lin.params['x']:+.3f}, balanced {lin.params['mention']:+.3f}"
    )

    text = "\n".join(out)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
