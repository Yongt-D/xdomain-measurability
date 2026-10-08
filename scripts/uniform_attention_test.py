# -*- coding: utf-8 -*-
"""
Forced-uniform a_k replacement test (pre-registration Appendix H). **Committed before any number of this test existed.**

Intervention (H.2): at inference, set PrototypeHead.support_temperature to 1e6 so that
support_scores = cos(q,s_k)/|T| fall into [-1e-6, 1e-6] and the softmax deviates from uniform by O(1e-6).
Everything else unchanged: same checkpoint, same source-validation frozen threshold, same support manifest, same full query set.
**src/gaplsegnet_v5_ch5.py is not modified** (section 4 item 6).

Criteria (H.3, fixed before running):
  U0 self-check : after the intervention mean H > 0.99999, otherwise the result is void
  U1 primary    : |IoU_uniform - IoU_base| < 0.0037 (smallest support-draw SD of arm C in stage 0b), for all three seeds
  U2            : report per seed the sign and size of delta, and the deviation of base's max_k a_k from 1/K
  U3 converse   : any seed with |delta| >= 0.0037 => the deviation carries performance; (3) must not be declared passed in reverse

H.5 exploratory: also report base's mean H at K in {1,4,8} (the K=1 entropy is skipped by definition); no criterion, no conclusion.
"""
import argparse, json, hashlib, sys, pathlib, time, math
import numpy as np, torch
from torch.utils.data import DataLoader
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import gaplsegnet_v5_ch5 as M
EPS = M.EPS
U1_THRESHOLD = 0.0037          # smallest support-draw SD of arm C in stage 0b (stage-0b calibration judgement, section 4)
U0_THRESHOLD = 0.99999
UNIFORM_TEMPERATURE = 1e6


def iou_of(c):
    return c[0] / (c[0] + c[1] + c[2] + EPS)


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ta = state['args']
    base = M.ABLATIONS[ta['ablation']]
    m = M.GAPLSegNetV5(base, pretrained=False, fusion_mode=ta['fusion_mode']).to(device)
    m.load_state_dict(state['model'])
    m.eval()
    return m, ta, state


def support_tensors(s_ds, s_names, man, device):
    pos = {n: i for i, n in enumerate(s_names)}
    pairs = [s_ds[pos[f]] for f in man['files']]
    return (torch.stack([p[0] for p in pairs], 0).to(device),
            torch.stack([p[1] for p in pairs], 0).to(device))


def run_pass(model, loader, s_img, s_msk, T, device, want_iou=True):
    """Return (tp,fp,fn) and the per-query normalised entropy / max a_k."""
    acc = [0, 0, 0]
    ent, mx = [], []
    with torch.no_grad():
        for qi, qm in loader:
            qi = qi.to(device, non_blocking=True)
            tgt = qm.to(device).long()
            B = qi.shape[0]
            si = s_img.unsqueeze(0).expand(B, *s_img.shape).contiguous()
            sm = s_msk.unsqueeze(0).expand(B, *s_msk.shape).contiguous()
            out = model(qi, si, sm)
            if want_iou:
                p = (out['predictions'] > T).long()
                acc[0] += int(((p == 1) & (tgt == 1)).sum())
                acc[1] += int(((p == 1) & (tgt == 0)).sum())
                acc[2] += int(((p == 0) & (tgt == 1)).sum())
            sa = out['support_attention']                       # (B, shots)
            shots = sa.shape[1]
            if shots > 1:
                h = -(sa * sa.clamp_min(EPS).log()).sum(1) / math.log(shots)
                ent.extend(h.tolist())
            mx.extend(sa.max(1).values.tolist())
    return acc, np.asarray(ent), np.asarray(mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--threshold', type=float, required=True)
    ap.add_argument('--target-view', default=str(ROOT / 'data_view' / 'inria'))
    ap.add_argument('--manifest', default=str(ROOT / 'results' / 'support_manifests' / 'inria_support_manifests.json'))
    ap.add_argument('--manifest-key', default='fmin0.01_K4_r0')
    ap.add_argument('--explore-keys', default='fmin0.01_K1_r0,fmin0.01_K8_r0',
                    help='H.5 exploratory: entropy only, no IoU')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=None)
    a = ap.parse_args()
    device = M.setup_device(a.gpu)
    T = a.threshold

    model, ta, state = load_model(a.checkpoint, device)
    orig_temp = float(model.prototype.support_temperature.detach().abs())
    print('loaded %s' % a.checkpoint)
    print('  ablation=%s fusion=%s seed=%s best_epoch=%s frozen threshold=%s'
          % (ta['ablation'], ta['fusion_mode'], ta['seed'], state['epoch'], T))
    print('  learned support_temperature |T| = %.6f' % orig_temp)

    tv = pathlib.Path(a.target_view)
    q_ds = M.BuildingDataset(tv / 'test' / 'image', tv / 'test' / 'label', train=False)
    s_ds = M.BuildingDataset(tv / 'train' / 'image', tv / 'train' / 'label', train=False)
    s_names = sorted(p.name for p in (tv / 'train' / 'image').iterdir())
    mans = json.loads(pathlib.Path(a.manifest).read_text(encoding='utf-8'))
    man = mans[a.manifest_key]
    K = man['K']
    s_img, s_msk = support_tensors(s_ds, s_names, man, device)
    loader = DataLoader(q_ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=True)

    t0 = time.time()
    acc_b, ent_b, mx_b = run_pass(model, loader, s_img, s_msk, T, device)
    iou_b = iou_of(acc_b)

    with torch.no_grad():
        model.prototype.support_temperature.fill_(UNIFORM_TEMPERATURE)
    acc_u, ent_u, mx_u = run_pass(model, loader, s_img, s_msk, T, device)
    iou_u = iou_of(acc_u)
    with torch.no_grad():
        model.prototype.support_temperature.fill_(orig_temp)

    delta = iou_u - iou_b
    u0 = bool(ent_u.size and ent_u.mean() > U0_THRESHOLD)
    u1 = bool(abs(delta) < U1_THRESHOLD)

    # ---- H.5 exploratory: entropy only ----
    explore = {}
    for key in [k for k in a.explore_keys.split(',') if k.strip()]:
        m2 = mans[key]
        ei, em = support_tensors(s_ds, s_names, m2, device)
        _, e_ent, e_mx = run_pass(model, loader, ei, em, T, device, want_iou=False)
        explore[key] = {
            'K': m2['K'],
            'entropy_mean': (float(e_ent.mean()) if e_ent.size else None),
            'entropy_sd': (float(e_ent.std(ddof=1)) if e_ent.size > 1 else None),
            'max_ak_mean': float(e_mx.mean()),
            'uniform_1_over_K': 1.0 / m2['K'],
        }

    res = {
        'checkpoint': str(a.checkpoint),
        'checkpoint_sha256': hashlib.sha256(open(a.checkpoint, 'rb').read()).hexdigest(),
        'train_args': ta, 'threshold': T, 'n_query': len(q_ds), 'K': K,
        'manifest_key': a.manifest_key, 'support_files': man['files'],
        'learned_support_temperature_abs': orig_temp,
        'uniform_temperature_used': UNIFORM_TEMPERATURE,
        'iou_base': iou_b, 'iou_uniform': iou_u, 'delta': delta,
        'entropy_base': {'mean': float(ent_b.mean()), 'sd': float(ent_b.std(ddof=1))},
        'entropy_uniform': {'mean': float(ent_u.mean()), 'sd': float(ent_u.std(ddof=1))},
        'max_ak_base': {'mean': float(mx_b.mean()), 'sd': float(mx_b.std(ddof=1)),
                        'min': float(mx_b.min()), 'max': float(mx_b.max())},
        'uniform_1_over_K': 1.0 / K,
        'U0_intervention_effective': u0,
        'U1_delta_below_noise_floor': u1,
        'U1_threshold': U1_THRESHOLD,
        'explore_H5': explore,
        'seconds': time.time() - t0,
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

    print('')
    print('  base    IoU = %.6f   H = %.6f' % (iou_b, ent_b.mean()))
    print('  uniform IoU = %.6f   H = %.8f' % (iou_u, ent_u.mean()))
    print('  delta = %+.6f   (U1 criterion |delta| < %.4f)' % (delta, U1_THRESHOLD))
    print('  base max_k a_k = %.6f +/- %.6f  [%.6f, %.6f]   uniform value 1/K = %.6f'
          % (mx_b.mean(), mx_b.std(ddof=1), mx_b.min(), mx_b.max(), 1.0 / K))
    print('    %s  U0 intervention effective (H > %.5f)' % ('pass' if u0 else '**FAIL, result void**', U0_THRESHOLD))
    print('    %s  U1 |delta| below the noise floor' % ('pass' if u1 else '**FAIL**'))
    for k, v in explore.items():
        print('  [H.5 exploratory] %s  K=%d  H=%s  max_ak=%.6f  1/K=%.6f'
              % (k, v['K'],
                 ('%.6f' % v['entropy_mean']) if v['entropy_mean'] is not None else 'n/a (K=1)',
                 v['max_ak_mean'], v['uniform_1_over_K']))
    print('  -> %s   (%.0fs)' % (out, res['seconds']))


if __name__ == '__main__':
    main()
