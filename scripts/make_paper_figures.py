# -*- coding: utf-8 -*-
"""
Paper figures 2 / 3 (data figures), all drawn from committed result files; no new number is produced.
  Fig. 2: C-A per seed -- Inria and Massachusetts, frozen threshold vs each arm's own oracle; +/-MDE reference band.
          Data: results/thrdiag_stage1_summary.json (Inria, K4_r0), results/thrdiag_mass_summary.json (Massachusetts, K4_r0),
                results/mass_summary.json (MDE_mass); Inria MDE = 0.0147 (stage-1 L1 judgement / scripts/mde_at_n.py).
  Fig. 3: pixel-level strata schematic + per-stratum dR (C-A): Inria f in {1,2,4,8} (results/rq5/*_gsd*.json) and Massachusetts (results/mass/*_K4_r0.json, 1 px band).
Usage: python scripts/make_paper_figures.py
"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'figures'; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(100, 112))
L = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
MDE_INRIA = 0.0147


def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    print('  ->', OUT / f'{name}.svg', '/ .png')


# ---------------- Fig. 2 ----------------
def fig2():
    th = L(ROOT / 'results' / 'thrdiag_stage1_summary.json')
    tm = L(ROOT / 'results' / 'thrdiag_mass_summary.json')
    mde_m = L(ROOT / 'results' / 'mass_summary.json')['main']['MDE_K4_CA']
    panels = [
        ('(a) Inria, 0.3 m (same GSD, different geography)', th['C_minus_A']['iou_frozen']['per_seed'], th['C_minus_A']['iou_oracle_wide']['per_seed'], MDE_INRIA),
        ('(b) Massachusetts, 1.0 m (natural GSD ratio 3.3, OSM labels)', tm['CA_frozen']['per_seed'], tm['CA_oracle']['per_seed'], mde_m),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, (title, fr, orc, mde) in zip(axes, panels):
        fr, orc = np.array(fr), np.array(orc)
        rng = np.random.default_rng(0)
        for x0, v, lab in ((0, fr, 'source-frozen threshold'), (1, orc, "each arm at its own target-oracle threshold")):
            jit = rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x0 + jit, v, s=16, color='#333', zorder=3)
            m = v.mean(); sd = v.std(ddof=1); half = 2.201 * sd / np.sqrt(len(v))   # t(0.975, 11)
            ax.errorbar(x0 + 0.3, m, yerr=half, fmt='s', color='#8a1f1f', ms=5, capsize=3, zorder=4)
            ax.text(x0 + 0.38, m, f'{m:+.3f}\n{int((v > 0).sum())}/12 > 0', va='center', fontsize=7.5, color='#8a1f1f')
        ax.axhspan(-mde, mde, color='#f0e6c8', alpha=0.9, zorder=0)
        ax.axhline(0, color='#777', lw=0.8)
        ax.set_xlabel(f'shaded band: ±MDE (n = 12) = {mde:.4f}' + (' (pilot-derived, pre-registered)' if mde == MDE_INRIA else ' (from the twelve-seed SD)'), fontsize=7, color='#8a6d1f')
        ax.set_xticks([0.15, 1.15]); ax.set_xticklabels(['frozen', 'oracle'])
        ax.set_xlim(-0.35, 1.65); ax.set_title(title, fontsize=8.5, loc='left')
    axes[0].set_ylabel('IoU difference, C − A (per training seed)')
    fig.text(0.5, -0.16, 'Dots: 12 training seeds (K = 4, support draw r0). Square: mean with descriptive 95 % interval. Shaded band: minimum detectable effect of a 12-seed comparison in that domain.\n'
             'Oracle thresholds use target labels and are a diagnostic upper bound only (Sections 4.1, 4.7, 4.8).', ha='center', fontsize=7, color='#444')
    save(fig, 'fig2_CA_frozen_vs_oracle')
    print(f'  fig2 numbers: Inria frozen {np.mean(panels[0][1]):+.4f} oracle {np.mean(panels[0][2]):+.4f}; Mass frozen {np.mean(panels[1][1]):+.4f} oracle {np.mean(panels[1][2]):+.4f}; MDE {MDE_INRIA} / {mde_m:.4f}')


# ---------------- Fig. 3 ----------------
S1, S2, S3, S4 = [0, 1], [2, 3], [4, 5], [6, 7]
BIG_EDGE, BIG_INT, NEAR, FAR = [4, 6], [5, 7], 8, 9


def recall(rows, strata, band=None):
    if band is None:
        n = np.array([r['n'] for r in rows]); p = np.array([r['p1'] for r in rows])
    else:
        n = np.array([r['alt'][band]['n'] for r in rows]); p = np.array([r['alt'][band]['p1'] for r in rows])
    return p[:, strata].sum() / max(n[:, strata].sum(), 1)


def strata_series(load_c, load_a):
    """Return dict: name -> per-seed array of dR (C-A) for S1..S4, big edge/int, near/far."""
    out = {k: [] for k in ('S1', 'S2', 'S3', 'S4', 'big_edge', 'big_int', 'near', 'far')}
    for s in SEEDS:
        rc, ra = load_c(s), load_a(s)
        for k, st in (('S1', S1), ('S2', S2), ('S3', S3), ('S4', S4), ('big_edge', BIG_EDGE), ('big_int', BIG_INT)):
            out[k].append(recall(rc, st) - recall(ra, st))
        for k, st in (('near', [NEAR]), ('far', [FAR])):
            out[k].append(recall(rc, st) - recall(ra, st))   # negative strata: p1/n = FPR
    return {k: np.array(v) for k, v in out.items()}


def fig3():
    series = {}
    for f in (1, 2, 4, 8):
        series[f'Inria f={f} ({0.3*f:.1f} m)'] = strata_series(
            lambda s, f=f: L(ROOT / 'results' / 'rq5' / f'Cconfidence_seed{s}_gsd{f}.json')['rows'],
            lambda s, f=f: L(ROOT / 'results' / 'rq5' / f'Aplain_seed{s}_gsd{f}.json')['rows'])
    series['Massachusetts 1.0 m (natural)'] = strata_series(
        lambda s: L(ROOT / 'results' / 'mass' / f'Cconfidence_seed{s}_fmin0.01_K4_r0.json')['rows'],
        lambda s: L(ROOT / 'results' / 'mass' / f'Aplain_seed{s}_fmin0.01_K4_r0.json')['rows'])

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.3)
    # (a) schematic
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 100, 60, facecolor='#eeeeee', edgecolor='none'))
    for (x, y, w, h, lab) in ((6, 6, 44, 30, 'S4 (large)'), (60, 8, 18, 14, 'S2'), (84, 40, 7, 6, 'S1'), (58, 34, 26, 20, 'S3')):
        ax.add_patch(Rectangle((x - 3, y - 3), w + 6, h + 6, facecolor='#d9d9d9', edgecolor='none'))   # near band
        ax.add_patch(Rectangle((x, y), w, h, facecolor='#9c9c9c', edgecolor='#333', lw=0.8))            # edge band
        if w > 8 and h > 8:
            ax.add_patch(Rectangle((x + 3, y + 3), w - 6, h - 6, facecolor='#555', edgecolor='none'))    # interior
        ax.text(x + w / 2, y + h / 2, lab, ha='center', va='center', fontsize=7, color='white' if w > 8 else '#111')
    for yy, col, lab in ((55, '#555', 'building interior (> band)'), (50, '#9c9c9c', 'building edge band (<= 4 px; Mass.: 1 px)'), (45, '#d9d9d9', 'background near band'), (40, '#eeeeee', 'background far')):
        ax.add_patch(Rectangle((2, yy - 2), 4, 3.5, facecolor=col, edgecolor='#333', lw=0.5)); ax.text(8, yy - 0.3, lab, fontsize=6.5, va='center')
    ax.set_title('(a) Label-defined strata', fontsize=8, loc='left')
    # (b) dR by size class
    ax = fig.add_subplot(gs[0, 1])
    names = list(series.keys()); cols = ['#bbbbbb', '#8a8a8a', '#555555', '#222222', '#8a1f1f']
    w = 0.16; xs = np.arange(4)
    for i, (nm, col) in enumerate(zip(names, cols)):
        vals = [series[nm][k].mean() for k in ('S1', 'S2', 'S3', 'S4')]
        ax.bar(xs + (i - 2) * w, vals, w, color=col, label=nm)
    ax.axhline(0, color='#777', lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(['S1 < 23 m²', 'S2 23–92', 'S3 92–369', 'S4 ≥ 369 m²'], fontsize=7)
    ax.set_ylabel('ΔRecall, C − A (mean of 12 seeds)'); ax.set_title('(b) Recall difference by instance size', fontsize=8, loc='left')
    ax.legend(fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
    # (c) big-building edge vs interior
    ax = fig.add_subplot(gs[1, 0]); xs = np.arange(len(names))
    e = [series[nm]['big_edge'].mean() for nm in names]; it = [series[nm]['big_int'].mean() for nm in names]
    cnt = [int((series[nm]['big_int'] < series[nm]['big_edge']).sum()) for nm in names]
    ax.bar(xs - 0.18, e, 0.34, color='#9c9c9c', label='edge band'); ax.bar(xs + 0.18, it, 0.34, color='#555', label='interior')
    for x, c in zip(xs, cnt): ax.text(x, max(e[x], it[x], 0) + 0.006, f'int<edge\n{c}/12', ha='center', fontsize=6, color='#333')
    ax.axhline(0, color='#777', lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(['Inria\nf=1', 'Inria\nf=2', 'Inria\nf=4', 'Inria\nf=8', 'Mass.\n1 m'], fontsize=7)
    ax.set_ylabel('ΔRecall, C − A, large buildings (S3 + S4)'); ax.set_title('(c) Large buildings: edge band vs interior', fontsize=8, loc='left'); ax.legend(fontsize=6.5, frameon=False)
    # (d) FPR near/far
    ax = fig.add_subplot(gs[1, 1])
    nr = [series[nm]['near'].mean() for nm in names]; fa = [series[nm]['far'].mean() for nm in names]
    ax.bar(xs - 0.18, nr, 0.34, color='#d9a441', label='near band'); ax.bar(xs + 0.18, fa, 0.34, color='#8a6d1f', label='far background')
    ax.axhline(0, color='#777', lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(['Inria\nf=1', 'Inria\nf=2', 'Inria\nf=4', 'Inria\nf=8', 'Mass.\n1 m'], fontsize=7)
    ax.set_ylabel('ΔFPR, C − A (mean of 12 seeds)'); ax.set_title('(d) False-positive rate difference in background strata', fontsize=8, loc='left'); ax.legend(fontsize=6.5, frameon=False)
    fig.text(0.5, 0.02, 'Inria: K = 4, support draw r0, 2 997 query patches, synthetic degradation f in {1, 2, 4, 8}. Massachusetts: 501 query patches, 1 px band. All values at the source-frozen threshold (Sections 4.6, 4.7).', ha='center', fontsize=7, color='#444')
    save(fig, 'fig3_strata')
    for nm in names:
        s = series[nm]
        print(f'  fig3 {nm}: S1 {s["S1"].mean():+.4f} S4 {s["S4"].mean():+.4f} [S4<S1 {int((s["S4"] < s["S1"]).sum())}/12] big edge {s["big_edge"].mean():+.4f} int {s["big_int"].mean():+.4f} near {s["near"].mean():+.4f} far {s["far"].mean():+.4f}')


if __name__ == '__main__':
    fig2(); fig3()
