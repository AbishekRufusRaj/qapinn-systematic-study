"""
render_tables.py -- generate the paper's data tables FROM the artefacts.

WHY THIS EXISTS
---------------
Every results table in PAPER.md was typed by hand and happened to agree with runs/.
`verify_paper.py` could tell you a number disagreed with disk, but only for the numbers
someone had remembered to assert, and it could not tell you that a table two sections
away still described a different sample size. Extending the arms from five seeds to
eight left Appendix B.1 and B.2 at n = 5, and section 8.3's inversion table quoting
n = 5 medians beside n = 8 tables, for an entire revision.

A generated table cannot be stale. This module renders each data table from runs/ in
exactly the markdown PAPER.md uses, then either checks or rewrites.

    python3 render_tables.py               # --check: report every stale table
    python3 render_tables.py --apply       # rewrite PAPER.md with rendered tables
    python3 render_tables.py --show NAME   # print one rendered table
    python3 render_tables.py --list

Tables are located by their HEADER ROW inside a named section, not by byte offset or by
their current contents, so editing the prose around a table does not break this and a
table whose contents drifted is still found and corrected.

Bolding is editorial -- it marks the best value or a significant result -- so each
builder applies its own rule explicitly. That keeps the emphasis meaningful rather than
dropping it to make generation easier.

NOT every table is generated. Theory, environment and instrument-validation tables
(sections 3, 5, 7, 10, C) are not derived from runs/ and are left alone; `--list` marks
which are covered.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

RUNS = "runs"
FAIL_LOSS = 1e-2
PAPER = "PAPER.md"


# ------------------------------------------------------------------ loading

def arm(pattern):
    """(errors, failed-flags, epochs, seeds) for a glob, ordered by seed."""
    e, f, ep, sd = [], [], [], []
    for d in sorted(glob.glob(os.path.join(RUNS, pattern)),
                    key=lambda p: int(re.search(r"_s(\d+)$", p).group(1))):
        mp, tp = os.path.join(d, "meta.json"), os.path.join(d, "metrics.json")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            continue
        m, t = json.load(open(mp)), json.load(open(tp))
        e.append(t["rel_l2"])
        f.append(float(m.get("final_total_loss", np.inf)) > FAIL_LOSS)
        ep.append(m.get("adam_epochs_used", float("nan")))
        sd.append(int(re.search(r"_s(\d+)$", d).group(1)))
    return np.array(e), np.array(f, bool), np.array(ep, float), sd


def pct(x, bold=False):
    s = f"{x * 100:.2f}%"
    return f"**{s}**" if bold else s


def row(cells):
    return "| " + " | ".join(cells) + " |"


# ----------------------------------------------------------------- builders

ARMS = [
    ("classical, gain 1.0", "conv_classical_s*"),
    ("classical, gain 0.5", "ctrl_classical_g0.5_s*"),
    ("classical, gain 2.0", "ctrl_classical_g2.0_s*"),
    ("classical, Xavier", "ctrl_classical_g1.7321_s*"),
    ("ring, default", "conv_ring_L2_s*"),
    ("ring, near-identity", "init_ring_L2_s*"),
    ("strong, default", "conv_strong_L2_s*"),
    ("strong, near-identity", "init_strong_L2_s*"),
    ("QCPINN (771)", "qc_qcpinn_s*"),
    ("Model-2 (2,751)", "qc_model2_s*"),
    ("Model-1 (7,851)", "qc_model1_s*"),
    ("heat classical", "heat_classical_s*"),
    ("heat ring", "heat_ring_L2_s*"),
    ("heat strong", "heat_strong_L2_s*"),
]


def t_gain_axis():
    spec = [("0.5", "ctrl_classical_g0.5_s*"),
            ("1.0 (PyTorch default)", "conv_classical_s*"),
            ("1.7321 (Xavier)", "ctrl_classical_g1.7321_s*"),
            ("2.0", "ctrl_classical_g2.0_s*"),
            ("2.8868 (Xavier × tanh gain)", "ctrl_classical_g2.8868_s*")]
    data = {lbl: arm(p) for lbl, p in spec}
    best = min(data, key=lambda k: np.median(data[k][0]))   # bold the best row
    out = [row(["gain", "median", "mean", "sd", "worst", "failures",
                "mean epochs to converge"]), "|---|---|---|---|---|---|---|"]
    for lbl, _ in spec:
        e, f, ep, _ = data[lbl]
        b = lbl == best
        out.append(row([lbl, pct(np.median(e), b), pct(e.mean(), b),
                        pct(e.std(ddof=1), b), pct(e.max(), b),
                        f"{int(f.sum())}/{len(e)}",
                        (f"**{np.nanmean(ep):,.0f}**" if b else f"{np.nanmean(ep):,.0f}")]))
    return out


def t_heat():
    spec = [("Classical", "heat_classical_s*"),
            ("QAPINN (strong, near-id)", "heat_strong_L2_s*"),
            ("QAPINN (ring)", "heat_ring_L2_s*")]
    data = {l: arm(p) for l, p in spec}
    best = min(data, key=lambda k: np.median(data[k][0]))
    out = [row(["model", "median", "mean", "sd", "worst", "failures"]),
           "|---|---|---|---|---|---|"]
    for l, _ in spec:
        e, f, _, _ = data[l]
        out.append(row([l, pct(np.median(e), l == best), pct(e.mean()),
                        pct(e.std(ddof=1)), pct(e.max()),
                        f"{int(f.sum())}/{len(e)}"]))
    return out


def _cell2x2(pat):
    e, f, _, _ = arm(pat)
    return [pct(np.median(e)), pct(e.mean()), pct(e.std(ddof=1)), pct(e.max()),
            f"{int(f.sum())}/{len(e)}"]


def t_2x2_strong():
    out = [row(["initialisation", "median", "mean", "sd", "worst", "failures"]),
           "|---|---|---|---|---|---|"]
    out.append(row(["default"] + _cell2x2("conv_strong_L2_s*")))
    out.append(row(["near-identity"] + _cell2x2("init_strong_L2_s*")))
    return out


def t_2x2_ring():
    out = [row(["initialisation", "median", "mean", "sd", "worst", "failures"]),
           "|---|---|---|---|---|---|"]
    out.append(row(["default"] + _cell2x2("conv_ring_L2_s*")))
    out.append(row(["near-identity"] + _cell2x2("init_ring_L2_s*")))
    return out


def t_inversion():
    """Section 8.3's statistic-dependence table -- the one that silently kept
    n = 5 numbers beside n = 8 tables."""
    def conv_only(pat):
        e, f, _, _ = arm(pat)
        keep = e[~f]
        return keep.mean() if len(keep) else float("nan")

    rows = []
    for stat, fn in [("all-seed mean", lambda p: arm(p)[0].mean()),
                     ("**median**", lambda p: float(np.median(arm(p)[0]))),
                     ("converged-only mean", conv_only)]:
        cells = [stat]
        # Emphasis marks the DEFAULT-init column in every row. That is where the
        # statistic-dependence lives: the all-seed mean puts ring ahead there while
        # the median and the converged-only mean put strong ahead, and the contrast
        # between those three cells is the whole point of the table. The
        # near-identity column agrees across all three statistics, so it carries no
        # emphasis.
        for init, rp, sp in [("default", "conv_ring_L2_s*", "conv_strong_L2_s*"),
                             ("near-identity", "init_ring_L2_s*", "init_strong_L2_s*")]:
            r, s = fn(rp), fn(sp)
            win = "ring" if r < s else "strong"
            lo, hi = (("ring", r), ("strong", s)) if r < s else (("strong", s), ("ring", r))
            mark = f"**{win}**" if init == "default" else win
            cells.append(f"{lo[0]} {pct(lo[1])} < {hi[0]} {pct(hi[1])} → {mark}")
        rows.append(row(cells))
    return [row(["statistic", "at default init", "at near-identity"]),
            "|---|---|---|"] + rows


def t_appendix_b1():
    ns = max(len(arm(p)[3]) for _, p in ARMS)
    hdr = ["configuration"] + [f"s{i}" for i in range(ns)]
    out = [row(hdr), "|" + "---|" * len(hdr)]
    for lbl, pat in ARMS:
        e, f, _, seeds = arm(pat)
        cells = [lbl]
        by = {s: (v, bad) for s, v, bad in zip(seeds, e, f)}
        for i in range(ns):
            if i in by:
                v, bad = by[i]
                cells.append(pct(v) + (" ✗" if bad else ""))
            else:
                cells.append("—")
        out.append(row(cells))
    return out


def t_appendix_b2():
    out = [row(["configuration", "median", "95% percentile bootstrap"]),
           "|---|---|---|"]
    rng = np.random.default_rng(20260921)
    for lbl, pat in ARMS:
        e, _, _, _ = arm(pat)
        if not len(e):
            continue
        boots = [float(np.median(rng.choice(e, len(e), replace=True)))
                 for _ in range(20000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(row([lbl, pct(float(np.median(e))), f"[{pct(lo)}, {pct(hi)}]"]))
    return out


TABLES = {
    "8.2-gain-axis":  ("8.2", "| gain | median | mean | sd | worst | failures | mean epochs to converge |", t_gain_axis),
    "8.3-strong":     ("8.3", None, t_2x2_strong),
    "8.3-ring":       ("8.3", None, t_2x2_ring),
    "8.3-inversion":  ("8.3", "| statistic | at default init | at near-identity |", t_inversion),
    "8.5-heat":       ("8.5", "| model | median | mean | sd | worst | failures |", t_heat),
    "B.1-per-seed":   ("B.1", None, t_appendix_b1),
    "B.2-bootstrap":  ("B.2", None, t_appendix_b2),
}


# ------------------------------------------------------------------ locating

def find_tables(text):
    """Every markdown table: (section, start_line, end_line, lines)."""
    lines = text.split("\n")
    out, sec, i = [], None, 0
    while i < len(lines):
        m = re.match(r"^#{2,4}\s+([A-Za-z0-9.]+)", lines[i])
        if m:
            sec = m.group(1).rstrip(".")
        if (lines[i].startswith("|") and i + 1 < len(lines)
                and set(lines[i + 1].replace("|", "").replace(" ", "")) <= set("-:")
                and lines[i + 1].strip()):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append((sec, i, j, lines[i:j]))
            i = j
            continue
        i += 1
    return out


def locate(text, name):
    sec, hdr, _ = TABLES[name]
    cands = [t for t in find_tables(text) if t[0] == sec]
    if hdr is not None:
        cands = [t for t in cands if t[3][0] == hdr]
    if not cands:
        return None
    if hdr is None:
        # disambiguate the two 2x2 tables by the label in the row above them
        want = "Strong" if name.endswith("strong") else "Ring"
        if name.startswith("8.3"):
            lines = text.split("\n")
            cands = [t for t in cands
                     if any(want in l for l in lines[max(0, t[1] - 3):t[1]])]
        else:
            cands = cands[:1]
    return cands[0] if cands else None


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--show")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("generated from runs/:")
        for k, (s, _, _) in TABLES.items():
            print(f"  {k:18} section {s}")
        return 0

    text = open(PAPER, encoding="utf-8").read()

    if a.show:
        print("\n".join(TABLES[a.show][2]()))
        return 0

    lines = text.split("\n")
    stale, edits = [], []
    for name in TABLES:
        loc = locate(text, name)
        if loc is None:
            print(f"  {name:18} NOT FOUND in {PAPER}")
            stale.append(name)
            continue
        _sec, i, j, cur = loc
        new = TABLES[name][2]()
        if cur == new:
            print(f"  {name:18} ok")
        else:
            print(f"  {name:18} STALE ({len(cur) - 2} rows on disk vs "
                  f"{len(new) - 2} rendered)")
            for c, n in zip(cur[2:], new[2:]):
                if c != n:
                    print(f"      paper : {c[:96]}")
                    print(f"      disk  : {n[:96]}")
                    break
            stale.append(name)
            edits.append((i, j, new))

    if a.apply and edits:
        for i, j, new in sorted(edits, reverse=True):
            lines[i:j] = new
        open(PAPER, "w", encoding="utf-8").write("\n".join(lines))
        print(f"\nrewrote {len(edits)} table(s) in {PAPER}")
        return 0

    if stale:
        print(f"\n{len(stale)} table(s) stale. Run with --apply to regenerate.")
        return 1
    print("\nEvery generated table matches the artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
