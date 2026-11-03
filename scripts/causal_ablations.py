# -*- coding: utf-8 -*-
"""
Mandatory causal ablations (1)-(5) of section 4 (mapping in pre-registration Appendix F.2). **Committed before any ablation number existed.**

Runs on the **stage-0b C+conf checkpoints** as a **diagnostic** (F.1); the formal ablations of the paper are re-run on the stage-1 checkpoints.

Criteria (F.2 verbatim, thresholds unchanged):
  (1) load the same checkpoint with geometry=False (DGM residuals of encoder2/3/4 switched off);
      the binary prediction must change in more than 0.5% of the pixels   [mapping corrected in Appendix G.2]
  (2a) support masks cyclically shifted by one -> target IoU must be **lower** than with matched supports
  (2b) all K supports taken from background-only patches (f<0.001) -> IoU must be **lower** than with matched supports
  (3) per-query normalised entropy H: mean in [0.05, 0.995] and SD(H) > 1e-4
  (4) per-image r_i = mean_l ||F'_l - F_l|| / ||F_l|| (l = DGM residuals of encoder2/3/4, query only):
      Var_i(r_i)/(mean_i r_i)^2 > 1e-8   [mapping corrected in Appendix G.2]
  (5) random-prototype control (direction from CPU seed 20260910, norm matched per batch): C+conf IoU must be **higher**

**One rule fixed before running** (stricter than F.2, never looser): each of the seed0/1/2 checkpoints is judged separately and
**a criterion passes only if all three seeds satisfy it**; one failing seed marks the criterion as failed.

Implementation constraint (F.2): only run-time hooks / a different AblationConfig; **src/gaplsegnet_v5_ch5.py is not modified**.
"""
import argparse, json, hashlib, sys, pathlib, time, math
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import gaplsegnet_v5_ch5 as M
EPS = M.EPS
RAND_PROTO_SEED = 20260910


def iou_of(c):
    return c[0] / (c[0] + c[1] + c[2] + EPS)


def make_random_proto_wrapper(head):
    """(5): call the original forward, then replace the fg/bg prototypes by fixed random vectors of matched norm and recompute prototype_logits."""
    orig = head.forward
    gen = torch.Generator(device='cpu').manual_seed(RAND_PROTO_SEED)

    def wrapper(query_features, support_features, support_mask, config):
        out = orig(query_features, support_features, support_mask, config)
        fg, bg = out['foreground_prototype'], out['background_prototype']
        rf = torch.randn(fg.shape, generator=gen).to(fg.device, fg.dtype)
        rb = torch.randn(bg.shape, generator=gen).to(bg.device, bg.dtype)
        rf = F.normalize(rf, p=2, dim=1, eps=EPS) * fg.norm(dim=1, keepdim=True)
        rb = F.normalize(rb, p=2, dim=1, eps=EPS) * bg.norm(dim=1, keepdim=True)
        pl, fs, bs = head.logits_from_prototypes(query_features, rf, rb)
        out = dict(out)
        out.update(prototype_logits=pl, foreground_similarity=fs, background_similarity=bs,
                   foreground_prototype=rf, background_prototype=rb)
        return out
    return wrapper


def load_model(ckpt_path, device, geometry=None):
    """AblationConfig field order = (name, geometry, prototype_mode, geometry_weighting) (Appendix G.1)."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ta = state['args']
    base = M.ABLATIONS[ta['ablation']]
    g = base.geometry if geometry is None else geometry
    cfg = M.AblationConfig(base.name, g, base.prototype_mode, base.geometry_weighting)
    m = M.GAPLSegNetV5(cfg, pretrained=False, fusion_mode=ta['fusion_mode']).to(device)
    m.load_state_dict(state['model'])
    m.eval()
    return m, ta, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--threshold', type=float, required=True, help='threshold selected on the source validation set and frozen')
    ap.add_argument('--target-view', default=str(ROOT / 'data_view' / 'inria'))
    ap.add_argument('--manifest', default=str(ROOT / 'results' / 'support_manifests' / 'inria_support_manifests.json'))
    ap.add_argument('--manifest-key', default='fmin0.01_K4_r0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=None)
    a = ap.parse_args()
    device = M.setup_device(a.gpu)
    T = a.threshold

    m_base, ta, state = load_model(a.checkpoint, device)
    m_goff, _, _ = load_model(a.checkpoint, device, geometry=False)
    m_rand, _, _ = load_model(a.checkpoint, device)
    m_rand.prototype.forward = make_random_proto_wrapper(m_rand.prototype)
    print('loaded %s' % a.checkpoint)
    print('  ablation=%s fusion=%s seed=%s best_epoch=%s  frozen threshold=%s'
          % (ta['ablation'], ta['fusion_mode'], ta['seed'], state['epoch'], T))
    print('  config: geometry=%s prototype_mode=%s geometry_weighting=%s'
          % (m_base.config.geometry, m_base.config.prototype_mode, m_base.config.geometry_weighting))

    # (4) hook the DGM modules and record the relative residual magnitude of the query forward pass
    dgm_rel = {}

    def mk_hook(tag):
        def hook(mod, inp, out):
            f = inp[0]
            rel = ((out - f).flatten(1).norm(dim=1)
                   / f.flatten(1).norm(dim=1).clamp_min(EPS))
            dgm_rel.setdefault(tag, []).append(rel.detach())
        return hook

    for tag in ('geometry2', 'geometry3', 'geometry4'):
        getattr(m_base, tag).register_forward_hook(mk_hook(tag))

    tv = pathlib.Path(a.target_view)
    q_ds = M.BuildingDataset(tv / 'test' / 'image', tv / 'test' / 'label', train=False)
    s_ds = M.BuildingDataset(tv / 'train' / 'image', tv / 'train' / 'label', train=False)
    s_names = sorted(p.name for p in (tv / 'train' / 'image').iterdir())
    pos = {n: i for i, n in enumerate(s_names)}
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding='utf-8'))[a.manifest_key]
    K = man['K']
    pairs = [s_ds[pos[f]] for f in man['files']]
    s_img = torch.stack([p[0] for p in pairs], 0).to(device)
    s_msk = torch.stack([p[1] for p in pairs], 0).to(device)

    # (2b): background-only supports (f < 0.001), deterministic selection
    t0 = time.time()
    fracs = []
    for i in range(len(s_names)):
        _, mk = s_ds[i]
        fracs.append(float((mk > 0.5).float().mean()))
    fracs = np.asarray(fracs)
    bg_idx = np.where(fracs < 0.001)[0]
    print('  background-only candidates %d/%d  scan %.0fs' % (len(bg_idx), len(s_names), time.time() - t0))
    assert len(bg_idx) >= K, 'fewer than K background-only patches'
    take = [int(bg_idx[round(j * (len(bg_idx) - 1) / max(K - 1, 1))]) for j in range(K)]
    bg_files = [s_names[i] for i in take]
    bpairs = [s_ds[i] for i in take]
    b_img = torch.stack([p[0] for p in bpairs], 0).to(device)
    b_msk = torch.stack([p[1] for p in bpairs], 0).to(device)
    print('  (2b) support = %s' % bg_files)

    s_msk_roll = torch.roll(s_msk, shifts=1, dims=0)      # (2a): image/mask mismatch

    loader = DataLoader(q_ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=True)
    VAR = ['base', 'geom_off', 'mismatch_a', 'mismatch_b', 'rand_proto']
    acc = {k: [0, 0, 0] for k in VAR}
    changed = 0
    total = 0
    ent, swm = [], []
    t0 = time.time()
    with torch.no_grad():
        for qi, qm in loader:
            qi = qi.to(device, non_blocking=True)
            tgt = qm.to(device).long()
            B = qi.shape[0]

            def rep(x):
                return x.unsqueeze(0).expand(B, *x.shape).contiguous()

            outs = {}
            outs['base'] = m_base(qi, rep(s_img), rep(s_msk))
            outs['geom_off'] = m_goff(qi, rep(s_img), rep(s_msk))
            outs['mismatch_a'] = m_base(qi, rep(s_img), rep(s_msk_roll))
            outs['mismatch_b'] = m_base(qi, rep(b_img), rep(b_msk))
            outs['rand_proto'] = m_rand(qi, rep(s_img), rep(s_msk))
            for k in VAR:
                p = (outs[k]['predictions'] > T).long()
                acc[k][0] += int(((p == 1) & (tgt == 1)).sum())
                acc[k][1] += int(((p == 1) & (tgt == 0)).sum())
                acc[k][2] += int(((p == 0) & (tgt == 1)).sum())
            pb = (outs['base']['predictions'] > T).long()
            pg = (outs['geom_off']['predictions'] > T).long()
            changed += int((pb != pg).sum())
            total += pb.numel()
            # (3) per-query normalised entropy
            sa = outs['base']['support_attention']                       # (B, shots)
            h = -(sa * sa.clamp_min(EPS).log()).sum(1) / math.log(sa.shape[1])
            ent.extend(h.tolist())
            # (4) relative DGM residual magnitude of the query forward pass of this batch (first dimension == B)
            per_stage = []
            for tag in ('geometry2', 'geometry3', 'geometry4'):
                qs = [t for t in dgm_rel.get(tag, []) if t.shape[0] == B]
                assert qs, 'hook did not capture the query forward pass: %s' % tag
                per_stage.append(qs[0])          # the query forward pass precedes the support forward pass
            swm.extend(torch.stack(per_stage, 0).mean(0).tolist())
            dgm_rel.clear()
    dt = time.time() - t0

    ent = np.asarray(ent)
    swm = np.asarray(swm)
    ious = {k: iou_of(acc[k]) for k in VAR}
    ratio = changed / total
    v4 = float(swm.var(ddof=1) / max(swm.mean() ** 2, 1e-30))
    res = {
        'checkpoint': str(a.checkpoint),
        'checkpoint_sha256': hashlib.sha256(open(a.checkpoint, 'rb').read()).hexdigest(),
        'train_args': ta, 'threshold': T, 'n_query': len(q_ds), 'K': K,
        'manifest_key': a.manifest_key, 'support_files': man['files'],
        'bg_support_files': bg_files, 'n_bg_candidates': int(len(bg_idx)),
        'rand_proto_cpu_seed': RAND_PROTO_SEED,
        'iou': ious,
        'ablation_1_changed_pixel_ratio': ratio,
        'ablation_3_entropy': {'mean': float(ent.mean()), 'sd': float(ent.std(ddof=1)),
                               'min': float(ent.min()), 'max': float(ent.max())},
        'ablation_4_relvar_across_images': v4,
        'ablation_4_dgm_rel_magnitude': {'mean': float(swm.mean()), 'sd': float(swm.std(ddof=1)),
                                         'min': float(swm.min()), 'max': float(swm.max())},
        'seconds': dt,
    }
    verdict = {
        '1_geometry_zeroed_changes_output': bool(ratio > 0.005),
        '2a_mask_mismatch_hurts': bool(ious['mismatch_a'] < ious['base']),
        '2b_background_support_hurts': bool(ious['mismatch_b'] < ious['base']),
        '3_attention_entropy_nondegenerate': bool(0.05 <= ent.mean() <= 0.995 and ent.std(ddof=1) > 1e-4),
        '4_geometry_signal_varies_across_images': bool(v4 > 1e-8),
        '5_beats_param_matched_random_prototype': bool(ious['base'] > ious['rand_proto']),
    }
    res['verdict'] = verdict
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

    print('')
    print('  IoU  base=%.6f  geom_off=%.6f  (2a)=%.6f  (2b)=%.6f  (5)rand=%.6f'
          % (ious['base'], ious['geom_off'], ious['mismatch_a'], ious['mismatch_b'], ious['rand_proto']))
    print('  (1) changed-pixel ratio = %.4f%%   (criterion > 0.5%%)' % (ratio * 100))
    print('  (3) normalised entropy mean = %.6f  SD = %.6e  [%.4f, %.4f]'
          % (ent.mean(), ent.std(ddof=1), ent.min(), ent.max()))
    print('  (4) relative DGM residual magnitude r: mean=%.6f sd=%.6e [%.5f, %.5f]'
          % (swm.mean(), swm.std(ddof=1), swm.min(), swm.max()))
    print('      Var_i(r_i)/mean^2 = %.6e   (criterion > 1e-8)' % v4)
    for k, v in verdict.items():
        print('    %s  %s' % ('pass' if v else '**FAIL**', k))
    print('  -> %s   (%.0fs)' % (out, dt))


if __name__ == '__main__':
    main()
