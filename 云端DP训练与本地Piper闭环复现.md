# 01｜云端训练 Diffusion Policy → 本地 RoboTwin 2.0 / Piper 单臂闭环评测

> **复现对象**：XPolicyLab 集成的图像 Diffusion Policy（DP），不是直接运行 Stanford Push-T 示例。云端 RTX 4090 负责训练；本地 Ubuntu Legion 已有 `RoboTwin`、`dp` 环境及 Piper 单臂仿真，不在本文重复安装。  
> **仓库基线**：`XPolicyLab` commit `34d6135d609d46a10ccb81da24d5ddad0b4c1f7a`（适配脚本针对该版本编写；升级代码须重新检查）。  
> **任务**：`piper_stack_red_green`，先搬绿色到基座，再将红色叠在绿色上。状态/动作均为 `[j1…j6, gripper]`，即 7D；观测为 `head_cam`、`left_cam` 两路 RGB 和 `agent_pos`。  
> **验证过的规模**：10 集 / 3,266 帧用于调通链路；60 集 / 20,127 帧用于正式训练。图像为 RGB `240×320`，Zarr 为 `C×H×W`，**不得再做 RGB↔BGR 交换**。

## 0. 所需文件与路径

本手册随附四个脚本（与本文件放在同一目录），复制到指定位置执行：

| 文件 | 用途 |
| --- | --- |
| `piper_m2_prepare.py` | 原 HDF5 → 双相机 7D Zarr；使用官方 `decode_image_bit()` |
| `setup_piper_dp_cloud.py` | 由原版 Dataset/YAML 生成 `PiperImageDataset`、`piper_single_task.yaml`（轻微通用化，移除了对 10 集 Zarr 名称的依赖） |
| `check_piper_dp_batch.py` | 10 集首次冒烟验证；固定测试 Batch 2 |
| `patch_piper_eval.py` | 在**本地已有 Piper 单臂执行器**上，给固定版本的 DP 推理适配器补两相机/7D 映射；会备份原代码，源码不匹配时停止 |

```bash
# 以下区分执行终端，不能把云端 /root 与本地 ~/robotics 混用
# 云端：
ROOT=/root/rivermind-data
DP=$ROOT/XPolicyLab/policy/DP
RAW=$ROOT/data/piper_record_one/piper_stack_red_green/piper
# 本地：
# LOCAL=$HOME/robotics/RoboTwin
```

## 1. 云端：建立 DP 环境（RTX 4090）

```bash
# 云端终端；已有可用 dp 环境可直接跳到第 2 节
mkdir -p /root/rivermind-data/{tmp,pip-cache,conda/envs}
export TMPDIR=/root/rivermind-data/tmp
export PIP_CACHE_DIR=/root/rivermind-data/pip-cache

conda config --prepend envs_dirs /root/rivermind-data/conda/envs
conda create -n dp python=3.10.22 -y
conda activate dp
python -m pip install --upgrade pip setuptools wheel

# CUDA 12.1 的 PyTorch 组合（不要求宿主机驱动显示相同 CUDA 版本）
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

cat > /root/rivermind-data/dp-constraints.txt <<'EOF'
numpy==1.23.5
numba==0.56.4
zarr==2.12.0
numcodecs==0.12.1
diffusers==0.11.1
moviepy==1.0.3
matplotlib==3.7.5
pandas==2.0.3
EOF

python -m pip install -c /root/rivermind-data/dp-constraints.txt \
  numpy numba zarr numcodecs diffusers hydra-core==1.3.2 \
  pandas opencv-python==4.10.0.84 pillow h5py scipy \
  huggingface_hub==0.25.2 \
  -i https://mirrors.aliyun.com/pypi/simple/

python -m pip install -c /root/rivermind-data/dp-constraints.txt \
  wandb ipdb gpustat dm_control moviepy imageio av matplotlib \
  termcolor sympy dill tqdm einops omegaconf \
  'PyYAML>=6' 'websockets>=14.0' 'msgpack>=1.0.8' \
  'msgpack-numpy>=0.4.8' 'pydantic>=2.5,<3' \
  -i https://mirrors.aliyun.com/pypi/simple/
```

下载并固定 XPolicyLab 版本（**云端不安装 RoboTwin 仿真**）：

```bash
cd /root/rivermind-data
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab
git checkout 34d6135d609d46a10ccb81da24d5ddad0b4c1f7a

a=$(git rev-parse HEAD); echo "XPolicyLab: $a"
conda activate dp
python -m pip install --no-deps -e ./policy/DP
python -m pip install --no-deps -e .
python -m pip check
python - <<'PY'
import torch, numpy, numba, zarr, diffusers, hydra, XPolicyLab, diffusion_policy
print('torch',torch.__version__, 'GPU',torch.cuda.is_available())
print('numpy',numpy.__version__,'numba',numba.__version__,'zarr',zarr.__version__)
print('diffusers',diffusers.__version__)
assert torch.cuda.is_available()
print('CLOUD DP ENV PASSED')
PY
```

**已遇到的安装坑**：Python 3.10 的实测环境使用 `numcodecs==0.12.1`，不要照旧错误约束 `0.15.1`；`moviepy` 固定 1.0.3；安装本地包用 `--no-deps`，避免重装已锁定的 NumPy/Numba/PyTorch。最终以 `pip check` 与真实导入测试为准。


## 2. 本地 Piper 单臂专家数据采集与上传

前提：本地 RoboTwin 2.0、Conda `RoboTwin` 环境及 Piper 单臂控制程序均已配置完成，任务 `piper_stack_red_green` 的专家测试已通过。

### 2.1 检查配置文件

本地项目目录：

```bash
conda activate RoboTwin
cd ~/robotics/RoboTwin

ls envs/piper_stack_red_green.py
ls env_cfg/task_config/piper_smoke.yml
ls env_cfg/task_config/piper_record_one.yml
```

配置说明：

| 文件 | 用途 | 是否必须 |
|---|---|---|
| `piper_smoke.yml` | Piper 单臂仿真测试配置 | 已有 `piper_record_one.yml` 时不需要 |
| `piper_record_one.yml` | 专家数据采集配置 | 必须 |
| `demo_clean.yml` | RoboTwin 原始配置模板 | 仓库已有，不直接用于单臂采集 |

如果已经有 `piper_record_one.yml`，直接进入 2.2。

如果只有 `piper_smoke.yml`，创建采集配置：

```bash
cd ~/robotics/RoboTwin

python - <<'PY'
from pathlib import Path
import yaml

src = Path("env_cfg/task_config/piper_smoke.yml")
dst = Path("env_cfg/task_config/piper_record_one.yml")

assert src.exists(), "缺少已验证的 Piper 单臂测试配置"

if dst.exists():
    print("Existing:", dst)
else:
    cfg = yaml.safe_load(src.read_text())
    cfg["episode_num"] = 10
    cfg["use_seed"] = False
    cfg["collect_data"] = True
    cfg["render_freq"] = 0
    cfg["save_freq"] = 15
    cfg["language_num"] = 1

    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Created:", dst)
PY
```

注意：必须沿用已通过仿真测试的 Piper 单臂配置，不要直接使用默认 ALOHA 双臂配置。

### 2.2 专家任务测试

如果本地保留了之前的测试脚本：

```bash
conda activate RoboTwin
cd ~/robotics/RoboTwin

python piper_pick_smoke.py \
  2>&1 | tee piper_pick_smoke.log
```

成功标志：

```text
Planner success: True
Task success: True
Joint path legal: True
EXPERT TEST PASSED
```

如果此前已经完成专家验证，可以直接进入数据采集。

### 2.3 采集前 10 集 HDF5

设置配置中的 `episode_num=10`：

```bash
cd ~/robotics/RoboTwin

python - <<'PY'
from pathlib import Path
import yaml

p = Path("env_cfg/task_config/piper_record_one.yml")
cfg = yaml.safe_load(p.read_text())

cfg["episode_num"] = 10
cfg["use_seed"] = False
cfg["collect_data"] = True
cfg["render_freq"] = 0

p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

开始采集：

```bash
conda activate RoboTwin
cd ~/robotics/RoboTwin

bash collect_data.sh \
  piper_stack_red_green \
  piper_record_one \
  0 \
  2>&1 | tee piper_collect_10.log
```

如果已经采集完成 10 集，则跳过本步骤，不要删除已有数据。

### 2.4 从 10 集续采到 60 集

首先备份配置和已有 seed：

```bash
cd ~/robotics/RoboTwin

cp env_cfg/task_config/piper_record_one.yml \
  env_cfg/task_config/piper_record_one.yml.bak

SEED=data/piper_record_one/piper_stack_red_green/piper/seed.txt

if [ -f "$SEED" ]; then
    cp "$SEED" "${SEED}.bak"
fi
```

将 `episode_num` 修改为 60：

```bash
python - <<'PY'
from pathlib import Path
import yaml

p = Path("env_cfg/task_config/piper_record_one.yml")
cfg = yaml.safe_load(p.read_text())

cfg["episode_num"] = 60
cfg["use_seed"] = False
cfg["collect_data"] = True
cfg["render_freq"] = 0

p.write_text(yaml.safe_dump(cfg, sort_keys=False))

print("Target episodes:", cfg["episode_num"])
PY
```

执行续采命令：

```bash
conda activate RoboTwin
cd ~/robotics/RoboTwin

bash collect_data.sh \
  piper_stack_red_green \
  piper_record_one \
  0 \
  2>&1 | tee piper_collect_60.log
```

`episode_num=60` 表示累计采集 60 集，已有 10 集时继续采集剩余 50 集。

不要删除已有 HDF5，也不要手工修改 `seed.txt`。

### 2.5 验证采集结果

数据目录：

```text
data/piper_record_one/piper_stack_red_green/piper/
├── seed.txt
├── data/
│   ├── episode_0000000.hdf5
│   ├── ...
│   └── episode_0000059.hdf5
└── video/
```

执行检查：

```bash
cd ~/robotics/RoboTwin

python - <<'PY'
from pathlib import Path
import h5py

root = Path("data/piper_record_one/piper_stack_red_green/piper")
files = sorted((root / "data").glob("episode_*.hdf5"))
seeds = [int(x) for x in (root / "seed.txt").read_text().split()]

assert len(files) == 60
assert len(seeds) == 60
assert len(set(seeds)) == 60

total_frames = 0

for i, p in enumerate(files):
    assert p.name == f"episode_{i:07d}.hdf5"

    with h5py.File(p, "r") as f:
        n = f["state/left_arm_joint_states"].shape[0]

        assert f["state/left_arm_joint_states"].shape == (n, 6)
        assert f["state/left_ee_joint_states"].shape == (n, 1)
        assert f["action/left_arm_joint_states"].shape == (n, 6)
        assert f["action/left_ee_joint_states"].shape == (n, 1)

        assert len(f["vision/cam_head/colors"]) == n
        assert len(f["vision/cam_left_wrist/colors"]) == n

        total_frames += n

print("Episodes:", len(files))
print("Total frames:", total_frames)
print("60 EPISODES HDF5 CHECK PASSED")
PY
```

本次实际项目的 60 集数据共有 20,127 帧。重新采集时，帧数可能因专家轨迹变化而不同。

### 2.6 打包并上传至云服务器

只打包 HDF5 和 seed，不需要上传 MP4：

```bash
cd ~/robotics/RoboTwin

tar -czf piper_60episodes_hdf5.tar.gz \
  -C data/piper_record_one/piper_stack_red_green/piper \
  data seed.txt
```

通过智川云 JupyterLab 或 SCP 上传至：

```text
/root/rivermind-data/piper_60episodes_hdf5.tar.gz
```

云服务器解压：

```bash
conda activate dp
cd /root/rivermind-data

mkdir -p \
  data/piper_record_one/piper_stack_red_green/piper

tar -xzf piper_60episodes_hdf5.tar.gz \
  -C data/piper_record_one/piper_stack_red_green/piper
```

检查数据数量：

```bash
find \
  data/piper_record_one/piper_stack_red_green/piper/data \
  -name "episode_*.hdf5" | wc -l
```

预期输出：

```text
60
```

完成后进入下一节：云端 HDF5 → Zarr 转换、Piper 两相机 7D DP 训练。


## 3. 云端：HDF5 → Zarr（两相机，7D）

```bash
conda activate dp
cd /root/rivermind-data
# piper_m2_prepare.py 已传到此目录
python piper_m2_prepare.py convert \
  --source /root/rivermind-data/data/piper_record_one/piper_stack_red_green/piper/data \
  --dest /root/rivermind-data/XPolicyLab/policy/DP/data/RoboTwin-piper_stack_red_green-piper_single-joint-60.zarr \
  --max-episodes 60 | tee piper_convert_60.log
```

**不要覆盖既有 Zarr**。转换器检测到目标目录已存在会主动拒绝；重做时先备份并确认目标，再处理。60 集使用独立 `-60.zarr`，原 10 集的 `RoboTwin-piper_stack_red_green-piper_single-joint.zarr` 保留。

```bash
python - <<'PY'
import zarr,numpy as np
p='/root/rivermind-data/XPolicyLab/policy/DP/data/RoboTwin-piper_stack_red_green-piper_single-joint-60.zarr'
g=zarr.open(p,mode='r')
ends=g['meta/episode_ends'][:]; n=int(ends[-1])
assert len(ends)==60 and n==20127 and np.all(np.diff(ends)>0)
for k,shape in {
 'head_camera':(n,3,240,320), 'left_camera':(n,3,240,320),
 'state':(n,7), 'action':(n,7)}.items():
    assert g['data/'+k].shape==shape,(k,g['data/'+k].shape)
print('60 EPISODES ZARR CHECK PASSED')
PY
```

转换关键点：`XPolicyLab.utils.process_data.decode_image_bit()` 输出即 RGB；状态与动作拼接顺序均为 `left_arm_joint_states (6) + left_ee_joint_states (1)`；原 HDF5 虽可能包含右侧同名字段，**本模型只用实际受控的左侧 7D**。

## 4. 云端：修改 DP Dataset、Hydra 任务和验证集划分

原版 `RobotImageDataset` 和在线 Runner 默认读取**头部＋左右腕部共三相机**；模型配置的动作也非 Piper 单臂 7D。随文脚本保留原文件，另建专用文件：

```bash
conda activate dp
cd /root/rivermind-data
python setup_piper_dp_cloud.py
# 应输出：PIPER DP CONFIG PREPARATION PASSED
```

新文件为：

```text
XPolicyLab/policy/DP/diffusion_policy/dataset/piper_image_dataset.py
XPolicyLab/policy/DP/diffusion_policy/config/task/piper_single_task.yaml
```

专用配置必须包含以下规格：

```yaml
shape_meta:
  obs:
    head_cam: {shape: [3, 240, 320], type: rgb}
    left_cam: {shape: [3, 240, 320], type: rgb}
    agent_pos: {shape: [7], type: low_dim}
  action: {shape: [7]}
# dataset._target_: diffusion_policy.dataset.piper_image_dataset.PiperImageDataset
# dataset.val_ratio: 0.1
```

**固定版本的验证集划分 Bug**：`diffusion_policy/common/sampler.py` 的 `get_val_mask` 原来把 `val_idxs` 写死为 `-1`，实际仅验证最后 1 集。正式训练前恢复按 `val_ratio` 随机选 episode：

```bash
cd /root/rivermind-data/XPolicyLab/policy/DP
python - <<'PY'
from pathlib import Path
p=Path('diffusion_policy/common/sampler.py'); s=p.read_text()
old='    val_idxs = -1'
new='    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)'
if old in s:
    p.with_suffix('.py.before_val_fix').write_text(s)
    p.write_text(s.replace(old,new,1))
elif new not in s:
    raise SystemExit('Unexpected sampler version: stop and inspect')
from diffusion_policy.common.sampler import get_val_mask
assert get_val_mask(60,0.1,seed=42).sum()==6
assert get_val_mask(10,0.1,seed=42).sum()==1
print('VALIDATION SPLIT PASSED (60: 54 train / 6 val)')
PY
```

不必额外手写 Hydra `${eval:...}` 解析器来检查。首次低成本 Batch 验证直接调用 Dataset：

```bash
conda activate dp
cd /root/rivermind-data/XPolicyLab/policy/DP
python - <<'PY'
import numpy as np, torch
from diffusion_policy.dataset.piper_image_dataset import PiperImageDataset
p='data/RoboTwin-piper_stack_red_green-piper_single-joint-60.zarr'
ds=PiperImageDataset(zarr_path=p,horizon=8,pad_before=2,pad_after=5,batch_size=2,val_ratio=0.1)
b=ds.postprocess(ds[np.array([0,1],dtype=np.int64)],torch.device('cpu'))
assert b['obs']['head_cam'].shape==(2,8,3,240,320)
assert b['obs']['left_cam'].shape==(2,8,3,240,320)
assert b['obs']['agent_pos'].shape==(2,8,7)
assert b['action'].shape==(2,8,7)
print('60 EPISODES PIPER BATCH CHECK PASSED')
PY
```

## 5. 云端：Smoke Test → 60 集正式训练

### 5.1 首次链路自检（已有通过结果可跳过）

使用 `training.debug=true`、小 Batch 和较少扩散推理步验证 DataLoader、网络前/反向和 checkpoint；**debug 模型不是正式可用模型**：

```bash
conda activate dp
cd /root/rivermind-data/XPolicyLab/policy/DP
export CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1
python train.py --config-name=robot_dp \
  task=piper_single_task bench_name=RoboTwin \
  task.dataset.zarr_path=data/RoboTwin-piper_stack_red_green-piper_single-joint-60.zarr \
  training.device=cuda:0 training.seed=42 training.resume=false \
  training.debug=true dataloader.batch_size=2 val_dataloader.batch_size=2 \
  policy.num_inference_steps=5 logging.mode=offline \
  hydra.run.dir=data/outputs/piper_smoke_60
```

**重要**：debug 与正式训练可能按 Zarr 名称＋seed 存到相同 checkpoint 目录。正式训练前检查 `find checkpoints -name '*.ckpt'`，将需要保留的 debug 产物移出正式目录；不要在已有正式权重时运行这个测试。

### 5.2 60 集正式复现实验：250 Epoch / Batch 64

```bash
conda activate dp
cd /root/rivermind-data/XPolicyLab/policy/DP
export CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1
set -o pipefail

python train.py \
  --config-name=robot_dp \
  task=piper_single_task \
  bench_name=RoboTwin \
  task.dataset.zarr_path=data/RoboTwin-piper_stack_red_green-piper_single-joint-60.zarr \
  training.device=cuda:0 \
  training.seed=42 \
  training.resume=false \
  training.debug=false \
  training.num_epochs=250 \
  training.max_train_steps=null \
  training.max_val_steps=null \
  training.checkpoint_every=50 \
  training.val_every=5 \
  training.sample_every=25 \
  training.lr_warmup_steps=500 \
  dataloader.batch_size=64 \
  dataloader.num_workers=0 \
  val_dataloader.batch_size=64 \
  val_dataloader.num_workers=0 \
  logging.mode=offline \
  hydra.run.dir=data/outputs/piper_dp_60_b64_ep250_seed42 \
  2>&1 | tee /root/rivermind-data/piper_dp_60_b64_ep250.log
```

这组实验约 **282 steps/epoch，70,500 次 optimizer steps**，模型约 107.33M 参数。文件保存在：

```text
checkpoints/RoboTwin-piper_stack_red_green-piper_single-joint-60-42/
  50.ckpt  100.ckpt  150.ckpt  200.ckpt  250.ckpt
```

运行记录的 250 Epoch 离线损失已基本收敛（此前分析报告：末期 train loss 约 `0.00055`，最低 val loss 约 `0.00135`）；**这不等价于 RoboTwin 闭环成功**，必须实际 rollout。保存多个中间 checkpoint，不默认 250 最优。

### 5.3 监控与已踩过的坑

```bash
# 新终端，不影响训练
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1
# 日志
cd /root/rivermind-data/XPolicyLab/policy/DP
tail -n 5 data/outputs/piper_dp_60_b64_ep250_seed42/logs.json.txt
find checkpoints/RoboTwin-piper_stack_red_green-piper_single-joint-60-42 -name '*.ckpt' -print
```

- 已出现 `num_workers=2` 在 `Training epoch 0: 0/282` 触发 **DataLoader worker Segmentation fault**；本复现固定 `num_workers=0`。单看 GPU-Util 0%–100% 波动不能证明训练异常。显存约 18.6 GiB（Batch 64），不要在正式训练中盲目提高 Batch。
- `robotworkspace.py` 的 `create_dataloader()` 虽接收 `pin_memory` 参数，固定版本却写死 `pin_memory=False`，先维持实测稳定配置，不把它误写成已开启。
- `PiperImageDataset` 使用自定义 BatchSampler；训练和验证的 Batch 都设为 **64**。遇到 `assert len(idx) == self.batch_size` 时先核查配置，不随意改 Dataset 逻辑。
- `training.resume=false` **一定是新训练**，不是续训；现有版本的恢复机制还涉及 `latest.ckpt`、优化器和 LR 调度，不能简单把 Epoch 250 改成 350 后重新启动。
- 无须创建指向同一 Zarr 的软链接；训练路径直接使用实际 `-60.zarr`，checkpoint 按该 basename 与 seed 区分 10 集、60 集。

## 6. 云端 checkpoint → 本地 RoboTwin

先在云端确认选择的权重存在；一般建议保留 150、200、250 以便同 seed 比较：

```bash
conda activate dp
cd /root/rivermind-data/XPolicyLab/policy/DP
RUN=RoboTwin-piper_stack_red_green-piper_single-joint-60-42
ls -lh checkpoints/$RUN/{150,200,250}.ckpt

# 打包 checkpoint、训练使用的专用配置及源码；不要打包庞大 Zarr
mkdir -p /root/rivermind-data/export
tar -czf /root/rivermind-data/export/piper_dp60_ckpt_bundle.tar.gz \
  checkpoints/$RUN \
  diffusion_policy/dataset/piper_image_dataset.py \
  diffusion_policy/config/task/piper_single_task.yaml \
  diffusion_policy/common/sampler.py
```

下载这个 tar.gz 到本地（JupyterLab 下载或 `scp`，地址因个人云实例而异）。**本地先检查文件来源和内容**，仅加载本人训练产生且可信的 `.ckpt`；PyTorch/dill 反序列化能够执行任意 Python 代码。

```bash
# 本地 Ubuntu
conda activate dp
cd ~/robotics/RoboTwin/XPolicyLab/policy/DP
# 把归档放到 ~/robotics/
tar -tzf ~/robotics/piper_dp60_ckpt_bundle.tar.gz | head
tar -xzf ~/robotics/piper_dp60_ckpt_bundle.tar.gz
find checkpoints/RoboTwin-piper_stack_red_green-piper_single-joint-60-42 -name '*.ckpt' -print
```

**不能只传 `250.ckpt` 而漏掉训练时的两相机 `PiperImageDataset` / Hydra task config**；本地 XPolicyLab 必须使用兼容版本（核对源码/commit）。

## 7. 本地：两相机 / 7D DP 在线推理适配

本地 `RoboTwin` 环境及 **Piper 单臂仿真执行器已安装并完成专家测试**；这里仅补策略侧和接口。原始 XPolicyLab 的 `model.py` 固定读取 `cam_right_wrist`，`DPRunner` 固定拼 `right_cam`；原 RoboTwin 通用 `xpolicylab_action_to_robotwin()` 会拼接双臂 14D。这三处不处理，**训练通过也无法正确闭环**。

先备份：

```bash
# 本地
cd ~/robotics/RoboTwin
cp XPolicyLab/policy/DP/model.py XPolicyLab/policy/DP/model.py.manual_backup
cp XPolicyLab/policy/DP/diffusion_policy/env_runner/dp_runner.py \
   XPolicyLab/policy/DP/diffusion_policy/env_runner/dp_runner.py.manual_backup
cp scripts/eval_policy_xpolicylab.py scripts/eval_policy_xpolicylab.py.manual_backup
```

把随文 `patch_piper_eval.py` 复制到 **本地 RoboTwin 根目录**执行（脚本还会单独生成 `.before_piper_eval` 备份）：

```bash
conda activate dp
cd ~/robotics/RoboTwin
python patch_piper_eval.py
python -m py_compile \
  XPolicyLab/policy/DP/model.py \
  XPolicyLab/policy/DP/diffusion_policy/env_runner/dp_runner.py \
  scripts/eval_policy_xpolicylab.py
```

脚本进行的明确修改：

1. `model.py`：删除 `right_cam`，将 RoboTwin 的 `left_arm_joint_state` / `left_ee_joint_state` 映射到单臂 `joint_state` / `ee_joint_state`，打包 7D `agent_pos`。
2. `dp_runner.py`：只构造 `head_cam`、`left_cam`、`agent_pos` 三个模型观测键。
3. `scripts/eval_policy_xpolicylab.py`：对于单臂动作字典，返回 7D `[j1…j6,gripper]` 而非追加右臂，**但只在你的本地 `take_action` 已支持 7D 时有效**。
4. 注册 `piper_single` action profile：`arm_dim: [6]`、`ee_dim: [1]`，保留原 `piper` 双臂 profile，不全局覆盖。

执行后先验证接口，不要跳过：

```bash
cd ~/robotics/RoboTwin
conda activate dp
python - <<'PY'
from XPolicyLab.utils.process_data import get_action_dim,get_robot_action_dim_info
print('piper_single dim:',get_action_dim('piper_single'))
print('profile:',get_robot_action_dim_info('piper_single'))
assert get_action_dim('piper_single')==7
print('7D PROFILE PASSED')
PY

# 查看本地已经调好的真实单臂执行器（先前记录采用 10 物理步插值）
grep -n -E 'def take_action|PIPER7|step_lim' envs/_base_task.py
# 如果 take_action 仍要求 14D，请先恢复你此前已通过的 Piper 7D 执行器补丁；不要直接跑策略。
```

注意：`patch_piper_eval.py` 针对前述 commit 和当前版本的 RoboTwin eval 客户端；**发现代码特征不匹配会停止**，不要强制替换。若你本地已经有能工作的专用推理/单臂补丁，应以那份实际验证的版本为准，不必二次打补丁。切勿将“原版 `embodiment: [piper]`”误认为已自动变成真实单臂控制器。

## 8. 本地：运行 DP → RoboTwin 闭环仿真

将 DP 部署配置切为单环境评测、按编号加载已传回的权重：

```bash
cd ~/robotics/RoboTwin/XPolicyLab/policy/DP
python - <<'PY'
from pathlib import Path
import yaml
p=Path('deploy.yml'); cfg=yaml.safe_load(p.read_text())
cfg['eval_batch']=False
cfg['checkpoint_num']=250  # 可改 150 / 200，不需重训
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('deploy.yml:',cfg['checkpoint_num'], 'eval_batch:',cfg['eval_batch'])
PY
```

本地任务配置使用已经经过 Piper 专家验证的 `piper_smoke.yml`（或你已有可视化评测专用 YAML），不能落回默认双臂 `demo_clean`。确认 `render_freq`、相机及成功判定；在评测环境脚本支持 `ROBOTWIN_EVAL_ARGS_FILE` 的版本执行：

```bash
cd ~/robotics/RoboTwin
cat > piper_dp_eval_args.txt <<'EOF'
--task_config
piper_smoke
--test_num
2
--instruction_type
seen
--expert_check
false
EOF

grep -q ROBOTWIN_EVAL_ARGS_FILE scripts/eval_policy.sh \
  || { echo '此版本不支持参数文件，请先查看脚本参数传递'; exit 1; }

conda activate dp
cd ~/robotics/RoboTwin/XPolicyLab/policy/DP
export EVAL_ENV_TYPE=sim
export ROBOTWIN_EVAL_ARGS_FILE="$HOME/robotics/RoboTwin/piper_dp_eval_args.txt"
set -o pipefail
bash eval.sh \
  RoboTwin \
  piper_stack_red_green \
  RoboTwin-piper_stack_red_green-piper_single-joint-60-42 \
  piper_single \
  joint \
  0 0 0 \
  dp RoboTwin \
  2>&1 | tee ~/robotics/piper_dp60_eval_250.log
```

这条命令含义依次为：`bench_name task_name ckpt_name env_cfg_type action_type seed policy_gpu env_gpu policy_conda_env sim_conda_env`。它启动**DP 策略服务＋RoboTwin 环境客户端**，`expert_check=false` 防止先用专家执行当前评测任务；它不是 `piper_pick_smoke.py` 的专家回放。官方本地同机评测接口及参数也见 [RoboTwin/XPolicyLab 文档](https://robotwin-platform.github.io/doc/usage/xpolicylab.html)。

结果查看：

```bash
cd ~/robotics/RoboTwin
find eval_result -type f \( -name '*.mp4' -o -name '_result.txt' \) | tail -n 30
grep -E 'Success!|Fail!|Final success rate|Policy rollout error' ~/robotics/piper_dp60_eval_250.log
```

**验收**：摄像头数据与关节状态进入策略；推理返回有限值的 `6` 个连续 7D 动作（实际以 checkpoint 的 `n_action_steps` 为准）；RoboTwin 的 7D `take_action` 确实执行；出现真实 `Success! / Fail!` 与评测视频。先跑 2 集确认接口，再固定未见过的 20–30 个 seeds 比较 `150.ckpt / 200.ckpt / 250.ckpt` 的成功率和失败阶段。

## 9. 实际故障排除速查

| 现象 | 优先处理 |
| --- | --- |
| `numcodecs` 依赖冲突 | 用已验证的 `0.12.1`，确认 Python 3.10 与 NumPy/Numba 锁定 |
| `Unsupported interpolation type eval` | 是独立 Hydra 检查代码没注册项目 resolver；使用 `train.py` 正常入口或直接验证 `PiperImageDataset` |
| `assert len(idx) == self.batch_size` | `dataloader` / `val_dataloader` 与自定义 Dataset Batch 机制不匹配；统一 Batch、检查源代码 |
| `nothing found at path ''` | Zarr 路径错误或虚构软链接；使用实际存在的 `.zarr` 路径 |
| `num_workers=2` Segmentation fault | 恢复 `dataloader.num_workers=0` 和 `val_dataloader.num_workers=0` |
| 离线 loss 很低，闭环抓错颜色/抓红后停滞 | 分辨**阶段识别不足**、动作误差累积、7D 执行器/采集控制频率不一致；不能靠追加 epoch 断言解决 |
| 到 1000 步提前结束 | `env_cfg/task_config/_eval_step_limit.yml` 中该任务可能缺省 1000；按实验需求单独设如 `piper_stack_red_green: 1500`，须与其他对照组相同 |
| 保存 MP4 模糊 | 默认视频可能只有 320×240；想提高清晰度需修改第三人称视频编码与分辨率，不需要改 DP 的输入尺寸 |
| `right_cam` 缺失 / 14D 对 7D | 检查 `model.py`、`dp_runner.py`、action profile、`xpolicylab_action_to_robotwin`、Piper `take_action()`；端到端均须 7D |

**复现完成标准**：60 集数据校验通过 → 250 Epoch/70,500 Step 权重生成 → checkpoint 与两相机适配传回本地 → DP 真正输出并执行 7D 动作 → 同 seed 仿真成功率和失败视频保存。只有训练 loss 收敛不算完成闭环评测。
