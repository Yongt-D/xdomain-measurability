# -*- coding: utf-8 -*-
"""
Paper figures 4-7, all drawn from committed result files (figure 7 runs an extra forward pass for the images only; it produces no number).
  Fig. 4: IoU(threshold) curves, Inria f=1 (A, C) and Massachusetts (A, B, C), 0.05-0.95; median frozen threshold, the protocol grid 0.40-0.60, sign flips of the C-A curve.
          Data: results/thrdiag_stage1/*_gsd1.json, results/thrdiag_mass/*.json.
  Fig. 5: seed-variance amplification: per-seed IoU of A/B/C in-domain (WHU test) / Inria / Massachusetts; SD and amplification factors.
          Data: results/stage1_train/*.results.json, results/stage1/*_K4_r*.json, results/mass/*_K4_r*.json.
  Fig. 6: synthetic degradation series: IoU(f) mean +/- SD per arm with per-seed thin lines; C-A against f with +/-MDE. Data: results/l2_summary.json.
  Fig. 7: qualitative comparison (A@frozen, C@frozen, A@oracle): two patches per domain chosen by a fixed rule (see pick_patches), not by hand.
Usage: python scripts/make_paper_figures2.py [--no-fig7] [--ckpt-dir DIR]
"""
import argparse, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'figures'; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(100, 112)); GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]
L = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'A': '#4a4a4a', 'B': '#2b6ca3', 'C': '#b0362b'}
LAB = {'A': 'A (baseline)', 'B': 'B (+geometry)', 'C': 'C (+geometry, +confidence-gated prototype)'}
TAG = {'A': 'Aplain', 'B': 'Bplain', 'C': 'Cconfidence'}


def save(fig, name):
    for ext, kw in (('svg', {}), ('png', {'dpi': 220}), ('pdf', {})):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', **kw)
    print('  ->', OUT / f'{name}.svg')


def mde(s_d, n=12):
    lo, hi = 1e-5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2; ncp = mid / (s_d / np.sqrt(n)); crit = stats.t.ppf(0.975, n - 1)
        p = stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp)
        if p >= 0.80: hi = mid
        else: lo = mid
    return float(hi)


# ---------------- Fig. 4: threshold curves ----------------
def fig4():
    def curves(files):
        return np.mean([[f['iou_by_threshold'][str(g)] for g in GRID] for f in files], 0), np.median([f['frozen_threshold'] for f in files]), np.median([f['oracle_threshold'] for f in files])
    inria = {a: curves([L(ROOT / 'results' / 'thrdiag_stage1' / f'{TAG[a]}_seed{s}_gsd1.json') for s in SEEDS]) for a in 'AC'}
    mass = {a: curves([L(ROOT / 'results' / 'thrdiag_mass' / f'{TAG[a]}_seed{s}.json') for s in SEEDS]) for a in 'ABC'}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for j, (title, D) in enumerate((('(a) Inria, 0.3 m', inria), ('(b) Massachusetts, 1.0 m', mass))):
        ax = axes[0, j]; ax.axvspan(0.40, 0.60, color='#eeeeee', zorder=0)
        for a, (c, ft, ot) in D.items():
            ax.plot(GRID, c, color=COL[a], lw=1.6, label=LAB[a])
            ax.axvline(ft, color=COL[a], lw=0.9, ls='--'); oi = int(np.argmin([abs(g - ot) for g in GRID])); ax.plot([GRID[oi]], [c[oi]], 'o', color=COL[a], ms=4)
        ax.set_title(title, loc='left', fontsize=8.5); ax.set_ylabel('IoU (mean of 12 seeds)')
        if j == 0: ax.legend(fontsize=7, frameon=False, loc='lower center')
        ax2 = axes[1, j]; d = D['C'][0] - D['A'][0]
        ax2.plot(GRID, d, color='#8a1f1f', lw=1.6); ax2.axhline(0, color='#777', lw=0.8); ax2.axvspan(0.40, 0.60, color='#eeeeee', zorder=0)
        ax2.fill_between(GRID, d, 0, where=d < 0, color='#8a1f1f', alpha=0.15)
        ax2.set_ylabel('C − A'); ax2.set_xlabel('decision threshold t')
        flips = [GRID[i] for i in range(1, len(GRID)) if np.sign(d[i]) != np.sign(d[i - 1])]
        ax2.text(0.06, ax2.get_ylim()[1] * 0.8, 'sign change at t ≈ ' + ', '.join(f'{f:.2f}' for f in flips) if flips else 'no sign change', fontsize=7.5)
    fig.text(0.5, -0.02, 'Solid: IoU of each arm against the decision threshold (mean over 12 seeds, K = 4, support draw r0). Dashed: median source-frozen threshold of that arm; dot: median target-oracle threshold (diagnostic only).\n'
             'Grey band: the pre-registered sensitivity grid 0.40–0.60. Lower panels: the C − A curve; shaded where negative.', ha='center', fontsize=7, color='#444')
    save(fig, 'fig4_threshold_curves')
    for nm, D in (('inria', inria), ('mass', mass)):
        print(f"  fig4 {nm}: frozen medians " + ' '.join(f"{a}={D[a][1]:.2f}" for a in D) + "; oracle medians " + ' '.join(f"{a}={D[a][2]:.2f}" for a in D))


# ---------------- Fig. 5: seed-variance amplification ----------------
def fig5():
    per = {}
    for a in 'ABC':
        ind = [L(ROOT / 'results' / 'stage1_train' / f'{TAG[a]}_seed{s}.results.json')['test_metrics']['iou'] for s in SEEDS]
        R = range(5) if a == 'C' else [0]
        inr = [np.mean([L(ROOT / 'results' / 'stage1' / f'{TAG[a]}_seed{s}_fmin0.01_K4_r{r}.json')['target']['iou'] for r in R]) for s in SEEDS]
        mas = [np.mean([L(ROOT / 'results' / 'mass' / f'{TAG[a]}_seed{s}_fmin0.01_K4_r{r}.json')['overall']['iou'] for r in R]) for s in SEEDS]
        per[a] = {'WHU test\n(in-domain)': np.array(ind), 'Inria\n(0.3 m)': np.array(inr), 'Massachusetts\n(1.0 m)': np.array(mas)}
    doms = list(per['A'].keys())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    rng = np.random.default_rng(0)
    for ax, dom in zip(axes, doms):
        for i, a in enumerate('ABC'):
            v = per[a][dom]; jit = rng.uniform(-0.15, 0.15, len(v))
            ax.scatter(i + jit, v, s=14, color=COL[a], zorder=3)
            ax.hlines(v.mean(), i - 0.28, i + 0.28, color=COL[a], lw=1.5)
            amp = v.std(ddof=1) / per[a][doms[0]].std(ddof=1)
            txt = f'SD {v.std(ddof=1):.4f}' + ('' if dom == doms[0] else f'\n×{amp:.1f}')
            ax.text(i, v.min() - 0.012 * (1 if dom == doms[0] else 4), txt, ha='center', va='top', fontsize=6.8, color=COL[a])
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['A', 'B', 'C']); ax.set_title(dom.replace('\n', ' '), fontsize=8.5, loc='left')
        lo = min(per[a][dom].min() for a in 'ABC'); hi = max(per[a][dom].max() for a in 'ABC'); pad = (hi - lo) * 0.6 + 0.01
        ax.set_ylim(lo - pad, hi + pad * 0.3)
    axes[0].set_ylabel('IoU at the source-frozen threshold')
    fig.text(0.5, -0.06, 'Each dot is one training seed (12 per arm; K = 4, C averaged over five support draws). Bars: mean. "×k": seed-to-seed SD relative to the in-domain SD of the same arm.', ha='center', fontsize=7, color='#444')
    save(fig, 'fig5_seed_variance')
    for a in 'ABC':
        print(f"  fig5 {a}: SD " + ' / '.join(f"{per[a][d].std(ddof=1):.4f}" for d in doms))


# ---------------- Fig. 6: degradation series ----------------
def fig6():
    S = L(ROOT / 'results' / 'l2_summary.json'); fs = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    for a in 'ABC':
        ps = np.array([S['main_curve'][f'{a}_gsd{f}']['per_seed'] for f in fs])      # 4 x 12
        for k in range(ps.shape[1]): ax.plot(fs, ps[:, k], color=COL[a], lw=0.5, alpha=0.35)
        ax.errorbar(fs, ps.mean(1), yerr=ps.std(1, ddof=1), color=COL[a], lw=1.8, marker='o', ms=4, capsize=3, label=LAB[a])
    ax.set_xscale('log', base=2); ax.set_xticks(fs); ax.set_xticklabels([f'f = {f}\n({0.3*f:.1f} m)' for f in fs]); ax.set_ylabel('IoU on Inria (source-frozen threshold)')
    ax.set_title('(a) All arms under synthetic down-sampling', loc='left', fontsize=8.5); ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    for key, lab, col in (('C_minus_A', 'C − A', '#8a1f1f'), ('B_minus_A', 'B − A', COL['B'])):
        d = [S['arm_diff'][f'{key}_gsd{f}'] for f in fs]
        eff = np.array([x['effect'] for x in d]); lo = np.array([x['ci95_descriptive'][0] for x in d]); hi = np.array([x['ci95_descriptive'][1] for x in d])
        ax.errorbar(fs, eff, yerr=[eff - lo, hi - eff], color=col, marker='s', ms=4, capsize=3, lw=1.5, label=lab)
        for f, x in zip(fs, d): ax.text(f, x['effect'], f"  {x['direction_positive']}/12", fontsize=6.5, color=col, va='bottom')
    m = [mde(S['arm_diff'][f'C_minus_A_gsd{f}']['sd_d']) for f in fs]
    ax.fill_between(fs, [-x for x in m], m, color='#f0e6c8', alpha=0.9, zorder=0, label='±MDE (n = 12)')
    ax.axhline(0, color='#777', lw=0.8); ax.set_xscale('log', base=2); ax.set_xticks(fs); ax.set_xticklabels([f'f = {f}' for f in fs])
    ax.set_ylabel('IoU difference'); ax.set_title('(b) Arm contrasts with descriptive 95 % intervals', loc='left', fontsize=8.5); ax.legend(fontsize=6.5, frameon=False)
    fig.text(0.5, -0.06, 'Thin lines: individual seeds. Numbers in (b): seeds with a positive difference. The MDE band is recomputed from the paired SD at each f.', ha='center', fontsize=7, color='#444')
    save(fig, 'fig6_degradation')
    print('  fig6 MDE by f:', [round(x, 4) for x in m])


# ---------------- Fig. 7: qualitative ----------------
def per_image_iou(rows):
    out = {}
    for r in rows:
        tp = sum(r['p1'][:8]); fn = sum(r['n'][:8]) - tp; fp = sum(r['p1'][8:]); pos = sum(r['n'][:8]); tot = sum(r['n'])
        out[r['name']] = (tp / max(tp + fp + fn, 1), pos / tot)
    return out


def pick_patches(dom_rows_A, dom_rows_C):
    """Fixed rule: among patches with a building fraction of 5-40%, take the median and the 90th-percentile patch of (IoU_C - IoU_A)."""
    A, C = per_image_iou(dom_rows_A), per_image_iou(dom_rows_C)
    cand = [(C[n][0] - A[n][0], n) for n in A if 0.05 <= A[n][1] <= 0.40]
    cand.sort()
    return [cand[len(cand) // 2][1], cand[int(len(cand) * 0.9)][1]], A, C


def fig7(ckpt_dir):
    import torch
    sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
    import gaplsegnet_v5_ch5 as M
    from model_loader import load_checkpoint
    device = M.setup_device(0)
    doms = [('Inria', ROOT / 'data_view_gsd' / 'inria_gsd1', ROOT / 'results' / 'support_manifests' / 'inria_support_manifests.json',
             ROOT / 'results' / 'rq5' / 'Aplain_seed100_gsd1.json', ROOT / 'results' / 'rq5' / 'Cconfidence_seed100_gsd1.json', ROOT / 'results' / 'thrdiag_stage1' / 'Aplain_seed100_gsd1.json'),
            ('Massachusetts', ROOT / 'data_view_mass', ROOT / 'results' / 'support_manifests' / 'mass_support_manifests.json',
             ROOT / 'results' / 'mass' / 'Aplain_seed100_fmin0.01_K4_r0.json', ROOT / 'results' / 'mass' / 'Cconfidence_seed100_fmin0.01_K4_r0.json', ROOT / 'results' / 'thrdiag_mass' / 'Aplain_seed100.json')]
    models = {a: load_checkpoint(pathlib.Path(ckpt_dir) / f'{TAG[a]}_seed100.pth', device)[0].eval() for a in 'AC'}
    rows = []
    for dom, view, man_p, ra, rc, thr in doms:
        RA, RC = L(ra), L(rc); names, A, C = pick_patches(RA['rows'], RC['rows'])
        ft_A, ft_C = RA['threshold'], RC['threshold']; ot_A = L(thr)['oracle_threshold']
        man = L(man_p)['fmin0.01_K4_r0']
        s_ds = M.BuildingDataset(view / 'train' / 'image', view / 'train' / 'label', train=False)
        s_names = sorted(p.name for p in (view / 'train' / 'image').iterdir()); pos = {n: i for i, n in enumerate(s_names)}
        pairs = [s_ds[pos[f]] for f in man['files']]
        si = torch.stack([p[0] for p in pairs]).unsqueeze(0).to(device); sm = torch.stack([p[1] for p in pairs]).unsqueeze(0).to(device)
        q_ds = M.BuildingDataset(view / 'test' / 'image', view / 'test' / 'label', train=False); q_names = sorted(p.name for p in (view / 'test' / 'image').iterdir())
        for nm in names:
            qi, qm = q_ds[q_names.index(nm)]
            with torch.no_grad():
                pa = models['A'](qi.unsqueeze(0).to(device), si, sm)['predictions'][0, 0].cpu().numpy()
                pc = models['C'](qi.unsqueeze(0).to(device), si, sm)['predictions'][0, 0].cpu().numpy()
            img = (qi.numpy().transpose(1, 2, 0) * np.array(M.STD) + np.array(M.MEAN)).clip(0, 1); gt = qm[0].numpy() > 0.5
            rows.append((f'{dom}: {nm}', img, gt, [(f'A @ frozen {ft_A:.2f}', pa > ft_A), (f'C @ frozen {ft_C:.2f}', pc > ft_C), (f'A @ oracle {ot_A:.2f} (diagnostic)', pa > ot_A)],
                         f'IoU A {A[nm][0]:.2f} / C {C[nm][0]:.2f}; building {A[nm][1]*100:.0f} %'))
    fig, axes = plt.subplots(len(rows), 5, figsize=(7.2, 1.55 * len(rows)))
    for i, (title, img, gt, preds, note) in enumerate(rows):
        dom_nm, fn = title.split(': '); axes[i, 0].imshow(img)
        axes[i, 0].set_title(dom_nm + chr(10) + fn.rsplit('.', 1)[0], fontsize=6, loc='left'); axes[i, 0].set_xlabel(note, fontsize=5.8, color='#444')
        axes[i, 1].imshow(gt, cmap='gray'); axes[i, 1].set_title('label', fontsize=6.5)
        for j, (lab, pr) in enumerate(preds):
            ov = np.zeros(gt.shape + (3,)); ov[pr & gt] = (0.2, 0.7, 0.3); ov[pr & ~gt] = (0.85, 0.2, 0.2); ov[~pr & gt] = (0.2, 0.3, 0.85); ov[~pr & ~gt] = 1
            axes[i, 2 + j].imshow(ov); axes[i, 2 + j].set_title(lab, fontsize=6.5)
            iou = (pr & gt).sum() / max((pr | gt).sum(), 1); axes[i, 2 + j].text(6, 40, f'IoU {iou:.2f}', fontsize=6.5, color='k', bbox=dict(fc='white', ec='none', pad=1))
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.005, 'Green: true positive; red: false positive; blue: false negative (seed 100, K = 4, r0). Patches are chosen by a fixed rule (building fraction 5–40 %; median and 90th-percentile C − A per-image IoU), not by hand.', ha='center', fontsize=6.5, color='#444')
    fig.subplots_adjust(hspace=0.55, wspace=0.05)
    save(fig, 'fig7_qualitative')
    print('  fig7 patches:', [r[0] for r in rows])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--no-fig7', action='store_true'); ap.add_argument('--ckpt-dir', default='ckpt/stage1'); a = ap.parse_args()
    fig4(); fig5(); fig6()
    if not a.no_fig7: fig7(a.ckpt_dir)
