#!/usr/bin/env python3
"""Create a separate 2-camera/7D Piper dataset and Hydra task config.

Place script under /root/rivermind-data and use the 'dp' conda environment. Does not
modify the upstream robot_image_dataset.py or default_task.yaml.
"""
from pathlib import Path
import re
import sys
import yaml

root = Path(__file__).resolve().parent
dp = root / 'XPolicyLab' / 'policy' / 'DP'
original = dp / 'diffusion_policy' / 'dataset' / 'robot_image_dataset.py'
task_source = dp / 'diffusion_policy' / 'config' / 'task' / 'default_task.yaml'
if not (original.is_file() and task_source.is_file()):
    sys.exit('Place this script in /root/rivermind-data and check XPolicyLab DP sources.')

src = original.read_text(encoding='utf-8')
class_old = 'class RobotImageDataset(BaseImageDataset):'
keys_old = 'keys=["head_camera", "left_camera", "right_camera", "state", "action"]'
if src.count(class_old) != 1 or src.count(keys_old) != 1:
    sys.exit('Unexpected upstream dataset version; stop and inspect it before patching.')
src = src.replace(class_old, 'class PiperImageDataset(BaseImageDataset):', 1)
src = src.replace(keys_old, 'keys=["head_camera", "left_camera", "state", "action"]', 1)
removed = []
remaining = []
for line in src.splitlines(keepends=True):
    strip = line.strip()
    if (strip.startswith('normalizer["right_cam"]')
            or strip.startswith('right_cam =')
            or strip.startswith("'right_cam':")
            or strip.startswith('"right_cam":')):
        removed.append(strip)
        continue
    remaining.append(line)
src = ''.join(remaining)
if len(removed) != 5 or 'right_camera' in src or re.search(r"['\"]right_cam['\"]", src):
    sys.exit('Unexpected camera references in upstream dataset; refused to write. Removed: ' + repr(removed))
compile(src, '<generated PiperImageDataset>', 'exec')

cfg = yaml.safe_load(task_source.read_text(encoding='utf-8'))
cfg['name'] = 'piper_stack_red_green'
cfg['shape_meta']['obs'] = {
    'head_cam': {'shape': [3, 240, 320], 'type': 'rgb'},
    'left_cam': {'shape': [3, 240, 320], 'type': 'rgb'},
    'agent_pos': {'shape': [7], 'type': 'low_dim'},
}
cfg['shape_meta']['action']['shape'] = [7]
cfg['dataset']['_target_'] = 'diffusion_policy.dataset.piper_image_dataset.PiperImageDataset'
cfg['dataset']['val_ratio'] = 0.1  # 10 episodes -> 1 val; 60 episodes -> 6 val (after sampler fix)
new_yaml = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
assert 'PiperImageDataset' in new_yaml and 'right_cam' not in new_yaml

def write_new(path: Path, contents: str):
    if path.exists():
        if path.read_text(encoding='utf-8') != contents:
            sys.exit(f'Existing file differs, not overwriting: {path}')
        print('UNCHANGED:', path)
    else:
        path.write_text(contents, encoding='utf-8')
        print('CREATED:', path)

write_new(dp / 'diffusion_policy' / 'dataset' / 'piper_image_dataset.py', src)
write_new(dp / 'diffusion_policy' / 'config' / 'task' / 'piper_single_task.yaml', new_yaml)
print('PIPER DP CONFIG PREPARATION PASSED')
print('Dataset Zarr path is selected when running train.py')
print('Next: test the dataset batch before starting GPU training.')
