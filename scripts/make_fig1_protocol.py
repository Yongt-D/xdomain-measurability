# -*- coding: utf-8 -*-
"""
Figure 1 of the paper: protocol overview drawn as a pictorial flowchart (vector PDF, matplotlib).
Every node is a glyph or a schematic mini-chart with a one-line label; the details are in the caption and in Section 3.
All mini-charts are SCHEMATIC (synthetic shapes, no result data). Usage: python scripts/make_fig1_protocol.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle, Circle, Arc
from scipy.ndimage import gaussian_filter

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'paper' / 'fig1_protocol_overview'
FS = 9.4
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': FS, 'pdf.fonttype': 42})
W, H = 10.0, 5.9
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
INK, GREY, BLUE, AMBER, GREEN, RED, LBLUE, LGREY = '#1a1a1a', '#5a5a5a', '#2f5d9b', '#c9a227', '#3f7d3f', '#b03a2e', '#dbe6f5', '#efefef'
rng = np.random.default_rng(3)


def inset(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([]); return a


def label(x, y, text, size=FS, bold=True, color=INK, ha='center', va='center', style='normal'):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, fontweight='bold' if bold else 'normal', color=color, style=style, linespacing=1.2)


def tag(x, y, t):
    ax.text(x, y, t, ha='right', va='top', fontsize=FS - 1.9, color=GREY, style='italic')


def arrow(p, q, ls='-', color=INK, lw=1.1, style='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=12, lw=lw, color=color, ls=ls, connectionstyle=style, shrinkA=2, shrinkB=2))


def frame(x, y, w, h, fc='white', ec=INK, ls='-', lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=ec, lw=lw, ls=ls))


def buildings(n, size, seed):
    r = np.random.default_rng(seed); img = np.zeros((size, size)) + 0.85
    for _ in range(n):
        w, h = r.integers(size // 12, size // 4, 2); x0, y0 = r.integers(0, size - w), r.integers(0, size - h)
        img[y0:y0 + h, x0:x0 + w] = 0.25 + 0.2 * r.random()
    return img


# ================= banner: sequential pre-registration =================
frame(0.25, H - 0.6, W - 0.5, 0.42, fc=LGREY, ec=GREY)
label(0.45, H - 0.39, 'P0  Sequential pre-registration', ha='left')
tl = inset(3.6, H - 0.55, 5.9, 0.32); tl.set_xlim(0, 27.5); tl.set_ylim(-1, 1); tl.axis('off')
tl.plot([0, 27], [0, 0], color=GREY, lw=1)
major = (0, 3, 6, 15, 17, 18, 21, 22, 24, 26)
for k in range(28): tl.plot([k, k], [-0.35, 0.35], color=BLUE if k in major else GREY, lw=1.6 if k in major else 0.8)
tl.text(0, 0.55, 'criteria & scripts committed before their numbers', fontsize=FS - 1.9, color=GREY, ha='left', va='bottom')
tl.text(27, -0.55, '27 dated revisions; each states what had already been seen', fontsize=FS - 1.9, color=GREY, ha='right', va='top')

# ================= row 1: source -> training -> checkpoints -> frozen threshold; ablations =================
y1 = H - 1.95; h1 = 1.05
frame(0.25, y1, 1.35, h1, fc=LBLUE)
im = inset(0.45, y1 + 0.32, 0.95, 0.55); im.imshow(buildings(14, 64, 1), cmap='gray', vmin=0, vmax=1, interpolation='nearest'); im.axis('off')
label(0.925, y1 + 0.17, 'Source domain', size=FS - 0.5)
frame(1.95, y1, 1.75, h1)
sa = inset(2.05, y1 + 0.32, 1.55, 0.6); sa.set_xlim(-1.2, 12); sa.set_ylim(-0.6, 2.6); sa.axis('off')
for j, (arm, c) in enumerate((('C', BLUE), ('B', GREY), ('A', INK))):
    sa.scatter(np.arange(12), [j] * 12, s=9, color=c); sa.text(-0.6, j, arm, ha='right', va='center', fontsize=FS - 1.5, color=c, fontweight='bold')
label(2.825, y1 + 0.17, '3 arms × 12 seeds', size=FS - 0.5)
frame(4.05, y1, 1.35, h1)
for k in (2, 1, 0):
    ax.add_patch(Rectangle((4.35 + 0.07 * k, y1 + 0.38 + 0.07 * k), 0.6, 0.45, fc='white', ec=INK, lw=0.9))
ax.text(4.65, y1 + 0.6, '#', ha='center', va='center', fontsize=12, color=BLUE, fontweight='bold')
label(4.725, y1 + 0.17, 'Checkpoints, SHA-256', size=FS - 0.5)
frame(5.75, y1, 1.9, h1); tag(7.6, y1 + h1 - 0.04, 'P3')
th = inset(5.95, y1 + 0.36, 1.5, 0.58); t = np.linspace(0.05, 0.95, 91); th.plot(t, 0.55 - 0.9 * (t - 0.47) ** 2, color=INK, lw=1.2)
th.axvspan(0.40, 0.60, color=AMBER, alpha=0.25); th.plot([0.47], [0.55], marker='o', color=BLUE, ms=4); th.plot([0.47, 0.47], [0.2, 0.55], color=BLUE, lw=0.8, ls=':')
th.set_xlim(0.05, 0.95); th.set_ylim(0.2, 0.62); th.text(0.5, 0.23, 'source val only', fontsize=FS - 2.1, color=GREY, ha='center')
for s in th.spines.values(): s.set_linewidth(0.5)
label(6.7, y1 + 0.17, 'Source-frozen threshold', size=FS - 0.5)
ax.add_patch(Rectangle((7.33, y1 + 0.78), 0.16, 0.12, fc=AMBER, ec=INK, lw=0.6)); ax.add_patch(Arc((7.41, y1 + 0.9), 0.1, 0.12, theta1=0, theta2=180, lw=0.8, color=INK))
frame(8.0, y1, 1.75, h1); tag(9.7, y1 + h1 - 0.04, 'P1')
ca = inset(8.15, y1 + 0.44, 1.45, 0.5); ca.set_xlim(0, 5.6); ca.set_ylim(0, 1); ca.axis('off')
for k in range(5):
    ca.add_patch(Circle((k + 0.8, 0.5), 0.34, fc='white', ec=INK, lw=0.8)); ca.text(k + 0.8, 0.5, '①②③④⑤'[k], ha='center', va='center', fontsize=FS - 0.5)
label(8.875, y1 + 0.17, 'Causal ablations', size=FS - 0.5); ax.text(8.875, y1 + 0.34, 'all five must pass', ha='center', va='center', fontsize=FS - 2.3, color=GREY)
for a, b in ((1.6, 1.95), (3.7, 4.05), (5.4, 5.75)): arrow((a, y1 + h1 / 2), (b, y1 + h1 / 2))
arrow((7.65, y1 + h1 / 2), (8.0, y1 + h1 / 2), ls='--', color=GREY)

# ================= row 2: targets + supports -> inference -> gates =================
y2 = H - 3.5; h2 = 1.2
frame(0.25, y2, 3.6, h2, fc=LBLUE); tag(3.8, y2 + h2 - 0.04, 'P5')
names = [('Inria', 18, 1.0, 0), ('blur ×2', 18, 2.2, 0), ('blur ×4', 18, 3.6, 0), ('Mass. 1 m', 7, 0.8, 5), ('sat. I', 10, 1.2, 7), ('sat. II', 12, 1.4, 9)]
for k, (nm, n, sig, sd) in enumerate(names):
    im = inset(0.4 + 0.56 * k, y2 + 0.42, 0.48, 0.48); img = buildings(n, 48, 11 + sd); im.imshow(gaussian_filter(img, sig), cmap='gray', vmin=0, vmax=1, interpolation='nearest'); im.axis('off')
    ax.text(0.64 + 0.56 * k, y2 + 0.36, nm, ha='center', va='top', fontsize=FS - 2.1, color=GREY)
label(2.05, y2 + 0.13, 'Target domains: same checkpoints, same thresholds', size=FS - 0.8)
frame(4.15, y2, 1.55, h2)
for k in range(4):
    im = inset(4.27 + 0.34 * k, y2 + 0.52, 0.3, 0.3); img = buildings(6, 32, 21 + k); im.imshow(gaussian_filter(img, 1.0), cmap='gray', vmin=0, vmax=1); im.axis('off')
    mk = inset(4.27 + 0.34 * k, y2 + 0.9, 0.3, 0.14); mk.imshow((img < 0.5)[:14, :], cmap='Blues', vmin=0, vmax=1.6, interpolation='nearest'); mk.axis('off')
label(4.925, y2 + 0.28, 'K-shot supports', size=FS - 0.5); label(4.925, y2 + 0.12, 'K∈{1,4,8}, 5 draws', size=FS - 2.1, bold=False, color=GREY)
frame(6.0, y2 + 0.2, 1.4, 0.8)
ax.add_patch(Polygon([(6.35, y2 + 0.45), (6.35, y2 + 0.75), (6.6, y2 + 0.6)], closed=True, fc=INK))
label(6.95, y2 + 0.6, 'Inference\nonly', size=FS - 0.5)
gx, gy = 8.45, y2 + 0.6
ax.add_patch(Polygon([(gx - 0.95, gy), (gx, gy + 0.62), (gx + 0.95, gy), (gx, gy - 0.62)], closed=True, fc='#fbf1d3', ec=AMBER, lw=1.1))
label(gx, gy + 0.12, 'Gates', size=FS - 0.3); ax.text(gx, gy - 0.16, 'provenance · machines · labels', ha='center', va='center', fontsize=FS - 2.7, color=GREY)
ax.text(gx + 1.0, gy - 0.05, '✗ stop', ha='left', va='center', fontsize=FS - 1.9, color=RED)
arrow((3.85, y2 + h2 / 2), (4.15, y2 + h2 / 2)); arrow((5.7, y2 + 0.6), (6.0, y2 + 0.6)); arrow((7.4, y2 + 0.6), (gx - 0.95, y2 + 0.6))
arrow((4.725, y1), (4.725, y2 + h2), ls='--', color=GREY)
ax.text(4.78, (y1 + y2 + h2) / 2, 'checkpoints re-used', fontsize=FS - 2.1, color=GREY, va='center', ha='left', style='italic')
arrow((6.7, y1), (6.7, y2 + 1.0), ls='--', color=GREY)
ax.text(6.75, (y1 + y2 + 1.0) / 2, 'frozen threshold', fontsize=FS - 2.1, color=GREY, va='center', ha='left', style='italic')
arrow((8.875, y1), (8.875, gy + 0.62), ls='--', color=GREY)
ax.text(0.3, y2 + h2 + 0.17, 'Identity control (P4): arms A and B never see the supports,' + chr(10) + 'so their outputs must be bit-identical across K and draws', fontsize=FS - 2.3, color=GREY, va='center', ha='left', style='italic')

# ================= row 3: four schematic analyses =================
y3 = 0.62; h3 = 1.45; xs = [0.25, 2.65, 5.05, 7.45]; w3 = 2.25
frame(xs[0], y3, w3, h3); tag(xs[0] + w3 - 0.05, y3 + h3 - 0.04, 'P2')
a1 = inset(xs[0] + 0.25, y3 + 0.42, 1.8, 0.85); d = rng.normal(0.02, 0.018, 12)
a1.axhspan(-0.015, 0.015, color=AMBER, alpha=0.25); a1.axhline(0, color=GREY, lw=0.6)
a1.scatter(np.arange(12), d, s=12, color=BLUE); a1.errorbar([13.2], [d.mean()], yerr=[[0.012], [0.012]], fmt='s', color=INK, ms=4, capsize=2, lw=1)
a1.set_xlim(-1, 14.5); a1.set_ylim(-0.045, 0.075); a1.text(5.5, 0.062, '12 seeds', fontsize=FS - 2.1, color=GREY, ha='center'); a1.text(13.2, -0.04, 'mean, CI', fontsize=FS - 2.3, color=GREY, ha='center'); a1.text(0, -0.013, '±MDE', fontsize=FS - 2.3, color='#8a6d1f', va='top')
for s in a1.spines.values(): s.set_linewidth(0.5)
label(xs[0] + w3 / 2, y3 + 0.2, 'Paired effect & MDE', size=FS - 0.5)

frame(xs[1], y3, w3, h3); tag(xs[1] + w3 - 0.05, y3 + h3 - 0.04, 'P3')
a2 = inset(xs[1] + 0.25, y3 + 0.42, 1.8, 0.85); t = np.linspace(0.05, 0.95, 91)
cA = 0.45 - 1.4 * (t - 0.12) ** 2; cC = 0.47 - 0.5 * (t - 0.42) ** 2
a2.plot(t, cA, color=INK, lw=1.1); a2.plot(t, cC, color=BLUE, lw=1.1); a2.axvspan(0.40, 0.60, color=AMBER, alpha=0.25)
a2.plot([0.48], [0.45 - 1.4 * (0.48 - 0.12) ** 2], 'o', color=INK, ms=4); a2.plot([0.12], [0.45], 'o', mfc='white', mec=INK, ms=4)
a2.plot([0.49], [0.47 - 0.5 * (0.49 - 0.42) ** 2], 'o', color=BLUE, ms=4); a2.plot([0.42], [0.47], 'o', mfc='white', mec=BLUE, ms=4)
a2.set_xlim(0.05, 0.95); a2.set_ylim(0.2, 0.52); a2.text(0.9, 0.23, 'threshold', fontsize=FS - 2.3, color=GREY, ha='right'); a2.text(0.07, 0.49, 'frozen ● vs oracle ○', fontsize=FS - 2.3, color=GREY)
for s in a2.spines.values(): s.set_linewidth(0.5)
label(xs[1] + w3 / 2, y3 + 0.2, 'Operating point', size=FS - 0.5)

frame(xs[2], y3, w3, h3); tag(xs[2] + w3 - 0.05, y3 + h3 - 0.04, 'P6')
a3 = inset(xs[2] + 0.3, y3 + 0.42, 1.7, 0.85); a3.set_xlim(0, 10); a3.set_ylim(0, 5); a3.axis('off')
a3.add_patch(Rectangle((0, 0), 10, 5, fc='#f4f1ea', ec='none'))
for (bx, by, bw, bh) in ((1.2, 1.0, 4.2, 3.0), (6.6, 2.4, 1.2, 1.0), (7.0, 0.6, 0.9, 0.9)):
    a3.add_patch(Rectangle((bx - 0.45, by - 0.45), bw + 0.9, bh + 0.9, fc='#e6d9b8', ec='none'))
    a3.add_patch(Rectangle((bx, by), bw, bh, fc='#b9c9e0', ec='none'))
    if bw > 2: a3.add_patch(Rectangle((bx + 0.5, by + 0.5), bw - 1.0, bh - 1.0, fc=BLUE, ec='none'))
a3.text(3.3, 2.5, 'interior', fontsize=FS - 2.3, color='white', ha='center', va='center'); a3.text(3.3, 4.55, 'edge', fontsize=FS - 2.1, color=INK, ha='center', va='center')
a3.text(9.0, 4.4, 'far', fontsize=FS - 2.1, color=GREY, ha='center'); a3.text(6.0, 0.35, 'near', fontsize=FS - 2.1, color='#8a6d1f', ha='center')
label(xs[2] + w3 / 2, y3 + 0.2, 'Pixel-level strata', size=FS - 0.5)

frame(xs[3], y3, w3, h3, fc=LGREY, ls='--'); tag(xs[3] + w3 - 0.05, y3 + h3 - 0.04, 'ext.')
a4 = inset(xs[3] + 0.6, y3 + 0.42, 1.45, 0.85); a4.set_xlim(0, 4); a4.set_ylim(0, 3); a4.axis('off')
cells = [['?', '?', '?', '?'], ['?', '?', '·', '·']]; cols = {'?': BLUE, '·': '#bbbbbb'}
for r_ in range(2):
    for c_ in range(4):
        a4.add_patch(Rectangle((c_, 1.9 - r_ * 1.0), 0.92, 0.92, fc='white', ec=GREY, lw=0.6)); a4.text(c_ + 0.46, 2.36 - r_ * 1.0, cells[r_][c_], ha='center', va='center', fontsize=FS + 1, color=cols[cells[r_][c_]], fontweight='bold')
a4.text(2.0, 0.55, 'resolved? sign? per pair' + chr(10) + '+ two more architectures', fontsize=FS - 2.4, color=GREY, ha='center', va='center')
a4.text(-0.1, 2.36, 'WHU→', fontsize=FS - 2.1, color=GREY, ha='right', va='center'); a4.text(-0.1, 1.36, 'Inria→', fontsize=FS - 2.1, color=GREY, ha='right', va='center')
label(xs[3] + w3 / 2, y3 + 0.2, 'External validity', size=FS - 0.5)
bus = y3 + h3 + 0.2
ax.plot([gx, gx], [gy - 0.62, bus], color=INK, lw=1.0); ax.plot([xs[0] + w3 / 2, xs[3] + w3 / 2], [bus, bus], color=INK, lw=1.0)
for x0 in xs: arrow((x0 + w3 / 2, bus), (x0 + w3 / 2, y3 + h3))
frame(0.25, 0.1, W - 0.5, 0.36, fc='#e6f2e6', ec=GREEN)
label(W / 2, 0.28, 'Five regularities R1–R5 with their boundaries', size=FS - 0.3)
for x0 in xs: arrow((x0 + w3 / 2, y3), (x0 + w3 / 2, 0.47))
fig.savefig(str(OUT) + '.pdf'); fig.savefig(str(OUT) + '_preview.png', dpi=110)
print('->', OUT.with_suffix('.pdf'))
