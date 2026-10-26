# -*- coding: utf-8 -*-
"""
Launch wrapper: set the torch.multiprocessing sharing strategy to file_system, then run the target script unchanged.
Purpose: containers where file-descriptor passing is unreliable (DataLoader workers fail with "Bad file descriptor").
Usage: python scripts/run_fs_sharing.py src/gaplsegnet_v5_ch5.py <original arguments...>
"""
import runpy, sys
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name='__main__')
