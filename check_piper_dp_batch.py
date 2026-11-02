#!/usr/bin/env python3
"""Run this from XPolicyLab/policy/DP in the dp conda environment."""
import numpy as np
import torch

from diffusion_policy.dataset.piper_image_dataset import PiperImageDataset

p = 'data/RoboTwin-piper_stack_red_green-piper_single-joint.zarr'
ds = PiperImageDataset(
    zarr_path=p, horizon=8, pad_before=2, pad_after=5,
    batch_size=2, val_ratio=0.0)
assert len(ds) >= 2, len(ds)
sample = ds[np.array([0, 1], dtype=np.int64)]
obs = ds.postprocess(sample, torch.device('cpu'))
for group_name in ('obs', 'action'):
    group = obs[group_name]
    if isinstance(group, dict):
        for k, v in group.items():
            print(f'{group_name}/{k}:', tuple(v.shape), v.dtype)
    else:
        print(f'{group_name}:', tuple(group.shape), group.dtype)
assert obs['obs']['head_cam'].shape == (2, 8, 3, 240, 320)
assert obs['obs']['left_cam'].shape == (2, 8, 3, 240, 320)
assert obs['obs']['agent_pos'].shape == (2, 8, 7)
assert obs['action'].shape == (2, 8, 7)
assert all(torch.isfinite(x).all() for x in (*obs['obs'].values(), obs['action']))
print('PIPER DP BATCH CHECK PASSED')
