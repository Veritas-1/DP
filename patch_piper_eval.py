#!/usr/bin/env python3
"""Patch the pinned XPolicyLab 34d6135 DP adapter for RoboTwin 7D/two-camera Piper.
Run from the local RoboTwin repository root. Backup copies are created. Stop on drift.
The local RoboTwin Piper 7D take_action implementation must already exist.
"""
from pathlib import Path
import json
import re
import py_compile

ROOT = Path.cwd()
DP = ROOT / 'XPolicyLab' / 'policy' / 'DP'
assert (ROOT / 'envs' / 'piper_stack_red_green.py').is_file(), 'Run from local RoboTwin root'
assert (DP / 'model.py').is_file(), 'XPolicyLab/policy/DP/model.py not found'


def update(path, transform):
    old = path.read_text(encoding='utf-8')
    new = transform(old)
    if old == new:
        print('UNCHANGED:', path)
        return
    backup = path.with_name(path.name + '.before_piper_eval')
    if not backup.exists():
        backup.write_text(old, encoding='utf-8')
    if path.suffix == '.py':
        compile(new, str(path), 'exec')
    path.write_text(new, encoding='utf-8')
    print('PATCHED:', path, '(backup:', backup.name + ')')


def model_patch(s):
    if 'PIPER7_OBS_PATCH' in s:
        return s
    assert s.count('def encode_obs(') == 1, 'Unexpected model.py encode_obs()'
    lines = s.splitlines(keepends=True)
    matches = [line for line in lines if 'right_cam =' in line or 'right_cam=right_cam' in line]
    assert len(matches) == 3, f'Expected 3 right-camera assignment lines, got {len(matches)}'
    lines = [line for line in lines if 'right_cam =' not in line and 'right_cam=right_cam' not in line]
    s = ''.join(lines)
    target = 'def encode_obs(observation, action_type, robot_action_dim_info):\n'
    assert s.count(target) == 1
    inject = '''    # PIPER7_OBS_PATCH: map RoboTwin left-arm observation to the single-arm profile.
    if len(robot_action_dim_info["arm_dim"]) == 1:
        state = observation["state"]
        if "joint_state" not in state:
            observation = dict(observation)
            observation["state"] = dict(state)
            observation["state"]["joint_state"] = state["left_arm_joint_state"]
            observation["state"]["ee_joint_state"] = state["left_ee_joint_state"]
'''
    s = s.replace(target, target + inject)
    assert 'right_cam' not in s, 'Residual right camera reference'
    return s


def runner_patch(s):
    if 'PIPER7_RUNNER_PATCH' in s:
        return s
    pattern = (r'(?m)^            obs_dict_input\["right_cam"\] = torch\.stack\(\n'
               r'                \[obs_torch\["right_cam"\] for obs_torch in obs_list_numpy\], dim=0\n'
               r'            \)\n')
    matches = list(re.finditer(pattern, s))
    assert len(matches) == 1, f'Expected 1 right_cam stack block, got {len(matches)}'
    s = re.sub(pattern, '            # PIPER7_RUNNER_PATCH: only head_cam and left_cam.\n', s)
    return s


def client_patch(s):
    if 'PIPER7_ACTION_PATCH' in s:
        return s
    anchor = '    robotwin_action_type = normalize_robotwin_action_type(str(resolved_action_type))\n'
    assert s.count(anchor) == 1, 'Unexpected xpolicylab_action_to_robotwin() implementation'
    addition = '''    # PIPER7_ACTION_PATCH: preserve 7D, do not append virtual right-arm actions.
    if robotwin_action_type == "qpos" and "joint_state" in action and "ee_joint_state" in action and "right_arm_joint_state" not in action:
        arm = np.asarray(action["joint_state"], dtype=np.float32).reshape(-1)
        grip = np.asarray(action["ee_joint_state"], dtype=np.float32).reshape(-1)
        assert arm.shape == (6,) and grip.shape == (1,), (arm.shape, grip.shape)
        return np.concatenate((arm, grip)), robotwin_action_type
'''
    return s.replace(anchor, anchor + addition)


update(DP / 'model.py', model_patch)
update(DP / 'diffusion_policy' / 'env_runner' / 'dp_runner.py', runner_patch)
update(ROOT / 'scripts' / 'eval_policy_xpolicylab.py', client_patch)

# The pinned policy helper reads env_cfg/<env_cfg_type>.yml and env_cfg/robot/_robot_info.json.
profile = ROOT / 'env_cfg' / 'piper_single.yml'
if not profile.exists():
    profile.write_text('config:\n  robot: piper_single\n', encoding='utf-8')
    print('CREATED:', profile)
else:
    assert 'piper_single' in profile.read_text(), 'Existing piper_single.yml inconsistent'

for path in (ROOT / 'env_cfg' / 'robot' / '_robot_info.json',
             ROOT / 'XPolicyLab' / 'utils' / 'robot' / '_robot_info.json'):
    if not path.exists():
        print('SKIPPED nonexistent profile registry:', path)
        continue
    info = json.loads(path.read_text(encoding='utf-8'))
    expected = {'arm_dim': [6], 'ee_dim': [1]}
    assert 'piper_single' not in info or info['piper_single'] == expected, f'Conflicting profile in {path}'
    if 'piper_single' not in info:
        backup = path.with_name(path.name + '.before_piper_eval')
        if not backup.exists():
            backup.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
        info['piper_single'] = expected
        path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + '\n', encoding='utf-8')
        print('REGISTERED piper_single 7D:', path)

print('PIPER DP EVAL ADAPTER PATCHED; validate 7D take_action and online observation/action shapes before rollout')
