# -*- coding: utf-8 -*-
"""
Unified loading of the two checkpoint kinds (Appendix W.2, "loader generalisation"). **Committed before any W number existed.**
  - GAPL (stage 1): the state has no 'arch' key -- the constructor call is **identical** to the original eval_crossdomain/eval_stratified/threshold_range_diag;
  - Appendix W architectures: state['arch'] in arch_baselines.ARCHS.
Returns (model, state, info); info['desc'] is a one-line description for logs, info['kind'] in {'gapl','arch'}.
"""
import pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import gaplsegnet_v5_ch5 as M


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    targs = state['args']
    if 'arch' in state:
        from arch_baselines import ArchSeg
        model = ArchSeg(state['arch'], pretrained=False).to(device)
        model.load_state_dict(state['model'])
        info = {'kind': 'arch', 'arch': state['arch'], 'seed': targs['seed'],
                'desc': f"arch={state['arch']} seed={targs['seed']}"}
    else:
        cfg = M.ABLATIONS[targs['ablation']]
        model = M.GAPLSegNetV5(cfg, pretrained=False, fusion_mode=targs['fusion_mode']).to(device)
        model.load_state_dict(state['model'])
        info = {'kind': 'gapl', 'arch': None, 'seed': targs['seed'],
                'desc': f"ablation={targs['ablation']} fusion={targs['fusion_mode']} seed={targs['seed']}"}
    return model, state, info
