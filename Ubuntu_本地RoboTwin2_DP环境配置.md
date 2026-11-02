# Ubuntu 22.04 从零搭建 RoboTwin 2.0 + Diffusion Policy：按实际成功流程修订版

> **文档版本**：2.0（2026-11-02 整理）  
> **依据**：原《Ubuntu 从零搭建 RoboTwin 2.0 + Diffusion Policy 复现实验环境》、本次用户实际终端日志，以及对应的项目源码。  
> **用途**：在一台**全新安装的 Ubuntu 22.04.5 LTS x86_64、带 NVIDIA RTX 4060（8GB）或相近显卡的笔记本/工作站**上，从零部署并验收。  
> **已实际走通的结果**：RoboTwin 专家示范成功并生成 MP4；XPolicyLab/DP 核心库与 GPU 检查成功；Stanford 官方 Push-T 预训练模型推理成功并生成 `eval_log.json` 和 MP4。  
> **尚未在这次会话中实际完成**：实验室服务器上的 RoboTwin DP 训练、RoboTwin 加载训练好的 DP checkpoint 后的在线闭环推理。相关内容仅作为后续流程，不能标记为已验证。  
> **重要**：以下流程根据一台成功机器的经验重新组织，但“在另一台全新机器上一次安装成功”仍取决于网络、显卡、驱动、镜像源和上游代码是否变化。每个阶段都设置验收点；**失败就停在当前阶段，不要带着错误继续**。

---

## 目录

0. [目标、环境划分和重要原则](#0-目标环境划分和重要原则)
1. [新系统硬件、系统与空间检查](#1-新系统硬件系统与空间检查)
2. [Ubuntu 基础依赖、Git 和文件夹](#2-ubuntu-基础依赖git-和文件夹)
3. [NVIDIA 驱动与 Vulkan](#3-nvidia-驱动与-vulkan)
4. [安装系统级 CUDA Toolkit 12.1](#4-安装系统级-cuda-toolkit-121)
5. [安装 Miniforge、Conda 和 Mamba](#5-安装-miniforgeconda-和-mamba)
6. [克隆 RoboTwin 官方仓库](#6-克隆-robotwin-官方仓库)
7. [安装和验收 RoboTwin 仿真环境](#7-安装和验收-robotwin-仿真环境)
8. [下载和验证 RoboTwin 三类资产](#8-下载和验证-robotwin-三类资产)
9. [RoboTwin GUI、专家示范与视频验收](#9-robotwin-gui专家示范与视频验收)
10. [创建 XPolicyLab 集成版 DP 环境](#10-创建-xpolicylab-集成版-dp-环境)
11. [创建独立 Stanford Push-T 环境](#11-创建独立-stanford-push-t-环境)
12. [下载预训练模型并完成一次真实推理](#12-下载预训练模型并完成一次真实推理)
13. [保存日志、版本和可复现证据](#13-保存日志版本和可复现证据)
14. [服务器训练与本地 RoboTwin DP 联合推理：后续可选](#14-服务器训练与本地-robotwin-dp-联合推理后续可选)
15. [最终验收清单与文件位置](#16-最终验收清单与文件位置)

---

## 0. 目标、环境划分和重要原则

### 0.1 三个 Conda 环境，不混用

| 环境 | Python | 主要软件 | 用途 | 本次实际状态 |
|---|---|---|---|---|
| `RoboTwin` | 3.10 | PyTorch 2.4.1 + cu121、NumPy 1.26.4、SAPIEN、MPLib、CuRobo | 本地机器人场景、GUI、专家示范和将来的在线评估 | **已验收** |
| `dp` | 3.10 | PyTorch 2.4.1 + cu121、NumPy 1.23.5、Numba 0.56.4、Diffusers 0.11.1 | RoboTwin/XPolicyLab 集成版 DP；未来加载服务器训练的权重 | **核心依赖已验收，尚未联合推理** |
| `dp-pusht` | 3.9 | PyTorch 2.4.1 + cu121、Gym 0.21、W&B 0.13.3、Stanford DP | 仅完成课程要求的 **Push-T 现成推理** | **已验收** |

**为什么要分开**：XPolicyLab/DP 与 Stanford 原版 DP 都有 Python 包 `diffusion_policy`，历史依赖和仿真 API 不完全相同。我们曾在 `dp` 里安装 Stanford Push-T 依赖，连续遇到 NumPy、Gym、W&B 和 Python 导入路径冲突，最终改成独立 `dp-pusht` 并成功推理。新机器应**从一开始就隔离**，不要重走混装过程。

### 0.2 实际设备和版本参考

- Ubuntu **22.04.5 LTS**，x86_64，内核 **6.8.0-138-generic**。
- NVIDIA GeForce **RTX 4060 Laptop GPU，8GB**；实际驱动版本 **580.178.04**。
- Miniforge 安装到 `~/miniforge3`；RoboTwin、Stanford 源码放在 `~/robotics`。
- **宿主机** CUDA Toolkit 安装到 `/usr/local/cuda-12.1`（通过 NVIDIA CUDA APT 仓库），不是同学文档所用的 Conda 内 CUDA 安装路线。
- RoboTwin 主仓库在成功环境中的 SHA：`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`。为减少新系统与上游更新导致的差异，本文优先固定到这个已知仓库版本，再记录实际更新后的子模块 SHA。
- Stanford 官方仓库在实际执行时未记录 Git SHA：**不能凭空捏造一个“已验证的 Stanford commit”**。安装后务必保存新机器克隆到的 SHA。若上游代码更新，检查本指南的脚本定位是否仍成立。

**显卡边界**：本文的 PyTorch 2.4.1 + cu121 是针对 RTX 4060 级别显卡已成功的组合；RTX 50/Blackwell 等较新显卡不能保证兼容，不建议不经验证直接照搬。其他 Ubuntu 版本也不能机械照抄下面的 CUDA Ubuntu 22.04 APT 仓库。

### 0.3 统一目录

```text
/home/<你的用户名>/
├── miniforge3/
│   └── envs/
│       ├── RoboTwin/
│       ├── dp/
│       └── dp-pusht/
└── robotics/
    ├── RoboTwin/                 # 官方 RoboTwin 源码 + XPolicyLab
    ├── diffusion_policy_official/ # Stanford 官方源码
    ├── dp_pusht_environment.yml # 专用 Push-T 环境文件
    └── experiment_logs/         # 所有安装/推理日志与版本快照
```

**不要在路径里使用中文**（用户名、目录也尽量用英文/ASCII）；不要根据其他同学使用的 Gitee 镜像目录 `script/`、`policy/` 去运行本指南的官方版路径 `scripts/`、`XPolicyLab/`。

---

## 1. 新系统硬件、系统与空间检查

打开 Ubuntu 终端（暂时没有 Conda）：

```bash
cat /etc/os-release
uname -m
uname -r
lspci | grep -Ei 'NVIDIA|VGA|3D'
free -h
df -h "$HOME"
command -v nvidia-smi || true
nvidia-smi || true
```

**验收**：Ubuntu 22.04、x86_64、检测到 NVIDIA GPU。RoboTwin 全量资产包括约 11GB 的背景 ZIP，还需要解压后的资源、多个 Conda 环境和可选数据；建议预留 **60GB 以上，最好 80GB 以上**，避免下载/解压时磁盘耗尽。这是推荐空间，不是官方硬门槛。

```bash
mkdir -p "$HOME/robotics/experiment_logs"
```

---

## 2. Ubuntu 基础依赖、Git 和文件夹

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl wget unzip zip ca-certificates \
  build-essential cmake ninja-build pkg-config patchelf \
  ffmpeg libgl1 libglib2.0-0 libegl1 libglfw3 \
  libosmesa6-dev libvulkan1 mesa-vulkan-drivers vulkan-tools \
  pciutils python3-pip python3-venv software-properties-common

git --version
ffmpeg -version | head -n 1
```

Ubuntu 未激活 Conda 时，有些机器只有 `python3`，没有 `python`：**不要为了这个错误重装 Python**。系统级脚本使用 `python3`；激活 Conda 后使用 `python`。

可选：若学校网络需要镜像，可以单独配置镜像；本文默认使用 GitHub / PyPI / conda-forge 官方地址，不假设任意镜像永久可用。

---

## 3. NVIDIA 驱动与 Vulkan

先检查：

```bash
nvidia-smi
```

- 能看到显卡、驱动版本，则**不重装驱动**。
- 全新系统没有驱动时，先使用 Ubuntu 推荐驱动：

```bash
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

重启后：

```bash
nvidia-smi
vulkaninfo --summary 2>&1 | tee "$HOME/robotics/experiment_logs/vulkaninfo_summary.txt"
```

**验收**：`vulkaninfo` 可列出 NVIDIA GeForce RTX GPU。它可能同时列出 AMD 核显或 llvmpipe；关键是 **NVIDIA Vulkan 设备确实存在**。偶尔出现非关键 ICD warning 并不必然失败。

`nvidia-smi` 顶部显示的 `CUDA Version: 13.0`（以本次成功设备为例）指**驱动支持的最大 CUDA 能力**，不等于已经安装的 CUDA Toolkit 版本。

---

## 4. 安装系统级 CUDA Toolkit 12.1

本次实际使用的是 **NVIDIA 官方 APT 仓库**安装 CUDA Toolkit 12.1，而不是在 Conda 里安装 CUDA 编译器：

```bash
wget -O /tmp/cuda-keyring_1.1-1_all.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1
```

添加到用户 `~/.bashrc`（**只添加一次**）：

```bash
cat >> "$HOME/.bashrc" <<'BASHRC'
# CUDA Toolkit 12.1 for RoboTwin
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
BASHRC
source "$HOME/.bashrc"
```

检查：

```bash
which nvcc
nvcc --version
ls /usr/local/cuda-12.1/include/cuda.h
nvidia-smi
```

**验收**：`nvcc` 报 CUDA 12.1，并且编译器头文件存在。**只安装 CUDA 版 PyTorch wheel 并不会提供 `nvcc`**。驱动、Toolkit、PyTorch 编译所用 CUDA 版本概念不同。

---

## 5. 安装 Miniforge、Conda 和 Mamba

```bash
mkdir -p "$HOME/Downloads"
cd "$HOME/Downloads"
curl -L -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
conda config --set auto_activate_base false
exec bash
```

在新终端检查：

```bash
conda --version
mamba --version
conda info --base
```

**验收**：`conda`、`mamba` 可用；Conda 根目录为 `~/miniforge3`。本文的独立环境由 `conda`/`mamba` 创建，不安装 Anaconda。

---

## 6. 克隆 RoboTwin 官方仓库

**使用你实际成功的官方 GitHub 仓库，包含 XPolicyLab 子模块**：

```bash
cd "$HOME/robotics"
git lfs install
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git RoboTwin
cd RoboTwin

# 固定到本次实际成功时的 RoboTwin 主仓库提交
# 全新机器执行该 checkout 是为了尽量避免 main 日后变动
# 如果此提交不再可用，停止并明确记录所用替代提交
# 安装脚本有可能把 XPolicyLab 更新到其最新 main

git checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
git submodule update --init --recursive

ls XPolicyLab/policy/DP
ls scripts/_install.sh
ls env_cfg/task_config/demo_clean.yml

git rev-parse HEAD
git submodule status | tee "$HOME/robotics/experiment_logs/robotwin_submodules_before.txt"
```

**注意**：仓库固定主提交，并不自动保证所有下载的 PyPI 包版本、外部源码依赖和 XPolicyLab 更新后子模块的状态完全固定。安装完成后应再次记录提交。**不要运行同学 Gitee 镜像的 `script/_install.sh`（单数目录）**。

---

## 7. 安装和验收 RoboTwin 仿真环境

```bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
python --version

cd "$HOME/robotics/RoboTwin"
set -o pipefail
bash scripts/_install.sh 2>&1 | tee "$HOME/robotics/experiment_logs/robotwin_install.log"
```

脚本会安装仿真 Python 依赖、处理 XPolicyLab 和 CuRobo 源码编译，并尝试安装 PyTorch3D。**本次实际成功的设备上 PyTorch3D 没有安装成功，但 2D RGB 仿真和 DP 相关的最小任务仍能运行**。因此：

- `pytorch3d FAILED` / `missing pytorch3d`：如果只做当前 2D RGB 示范任务，可暂时记录为可选缺失。
- `sapien`、`mplib`、`curobo` 任一无法导入，或 CuRobo 编译失败：不能继续宣称仿真环境成功。
- 脚本最后显示 `Installation basic environment complete!` **不是唯一成功标准**；必须做下面的检查。

```bash
conda activate RoboTwin
python - <<'PY'
import torch, numpy
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('NumPy:', numpy.__version__)
assert torch.cuda.is_available(), 'PyTorch cannot access GPU'
print('GPU:', torch.cuda.get_device_name(0))
for name in ('sapien', 'mplib', 'curobo'):
    __import__(name)
    print(name, 'OK')
try:
    import pytorch3d
    print('pytorch3d OK')
except ImportError:
    print('pytorch3d unavailable (optional for this 2D task)')
PY
python -m pip check
```

本次机器曾验证的核心版本：`torch 2.4.1+cu121`、`numpy 1.26.4`、`warp-lang 1.12.0`、`setuptools 69.5.1`。检查新系统的实际输出，必要时定位依赖冲突，**不要不看错误就重复执行整份安装脚本**（CuRobo 的源码目录可能已存在）。

安装后记录：

```bash
cd "$HOME/robotics/RoboTwin"
{
  git rev-parse HEAD
  git submodule status
  git status --short
} | tee "$HOME/robotics/experiment_logs/robotwin_git_after_install.txt"
```

---

## 8. 下载和验证 RoboTwin 三类资产

本次成功机器使用官方资产仓库：Hugging Face 数据集 `TianxingChen/RoboTwin2.0`，包含：

1. `embodiments.zip`：机器人本体（含 Aloha、Piper 等）。
2. `objects.zip`：可操作物体。
3. `background_texture.zip`：背景纹理。文件较大，约 11GB；`demo_randomized` 等随机背景配置会用到。

**本次曾踩坑**：直接运行 `bash scripts/_download_assets.sh` 时，背景 ZIP 传输中断，脚本仍继续解压另外两份资产，最终仿真报 `./assets/background_texture/seen` 不存在。新机建议按下列**逐个下载、逐个校验、逐个解压**的方式，错误时立即停止，尽量避免“表面结束，实际缺资源”。

```bash
conda activate RoboTwin
cd "$HOME/robotics/RoboTwin"
export HF_HUB_DOWNLOAD_TIMEOUT=120

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
from zipfile import ZipFile

assets = Path.cwd() / 'assets'
assets.mkdir(parents=True, exist_ok=True)
for filename in ('embodiments.zip', 'objects.zip', 'background_texture.zip'):
    print('\nDownloading:', filename, flush=True)
    path = hf_hub_download(
        repo_id='TianxingChen/RoboTwin2.0',
        repo_type='dataset',
        filename=filename,
        local_dir=str(assets),
    )
    print('Validating:', path, flush=True)
    with ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f'Invalid ZIP member: {bad}')
        print('Extracting:', filename, flush=True)
        archive.extractall(assets)
print('All three archives validated and extracted.')
PY

# 将 CuRobo 配置中的资源路径更新为实际本机路径
python scripts/update_embodiment_config_path.py

# 三个检查必须都能看到目录
ls -ld assets/embodiments assets/objects assets/background_texture/seen
```

**处理下载中断**：只重新运行上面的下载代码，Hugging Face Hub 通常会利用可复用的本地缓存/未完成记录；也可以单独用 `hf_hub_download` 下载缺失的 ZIP，再 `unzip -tq`、`unzip -q -o ... -d assets`。不要因为其中一份 ZIP 失败就删掉已经验证成功的其他资源。磁盘不足时可以在验证、解压后删除 ZIP（保留已解压资产）。

如果你选择只做 `demo_clean`、不做随机背景，也可以先下载 `embodiments.zip` 和 `objects.zip`，但**本指南的完整资源验收要求三类都在**，这与你实际成功机器一致。

---

## 9. RoboTwin GUI、专家示范与视频验收

### 9.1 首先区分：专家示范 **不是** DP 推理

`collect_data.sh` 运行的是 RoboTwin 的专家动作/数据采集流程；即使视频里机械臂完成任务，也不能当作扩散策略神经网络推理。这一阶段验收的是 **SAPIEN + Vulkan + CuRobo + 资产 + 相机/视频**。

### 9.2 用你的实际成功配置做最小 GUI 验收

本次实际成功测试采用：基于 `demo_clean.yml` 创建 `demo_preview.yml`，开启 `render_freq: 20`，只需要 1 条成功示范（可能尝试多个种子）。

```bash
conda activate RoboTwin
cd "$HOME/robotics/RoboTwin"

cp env_cfg/task_config/demo_clean.yml env_cfg/task_config/demo_preview.yml
sed -i 's/^render_freq: 0$/render_freq: 20/' env_cfg/task_config/demo_preview.yml
sed -i 's/^episode_num: 50$/episode_num: 1/' env_cfg/task_config/demo_preview.yml

grep -E '^(render_freq|episode_num|eval_video_log):' env_cfg/task_config/demo_preview.yml

set -o pipefail
bash collect_data.sh beat_block_hammer demo_preview 0 \
  2>&1 | tee "$HOME/robotics/experiment_logs/robotwin_preview.log"
```

**预期现象**：SAPIEN GUI 可能弹出；运动规划阶段窗口有时显示“无响应”甚至随后消失，不一定是错误。**以终端是否出现 `simulate data episode 0 success!`、是否正常完成采集和是否生成 MP4 为准**。

本次成功机器的真实记录（**只是示例结果，新机器 seed/帧数不保证相同**）：

```text
simulate data episode 0 fail! (seed = 0)
simulate data episode 0 success! (seed = 1)
Complete simulation, failed 1 times / 2 tries
[Start Data Collection]
Video is saved to ./data/demo_preview/beat_block_hammer/aloha_agilex/video/episode_0000000.mp4
```

查找视频及示范数据：

```bash
cd "$HOME/robotics/RoboTwin"
find data/demo_preview -type f -name '*.mp4'
find data/demo_preview -type f -name '*.hdf5'

# 若已找到下面路径，直接播放：
ffplay data/demo_preview/beat_block_hammer/aloha_agilex/video/episode_0000000.mp4
```

如果不需要 GUI，只要官方默认配置的采集验证，原始指南使用：

```bash
bash collect_data.sh beat_block_hammer demo_randomized 0 \
  2>&1 | tee "$HOME/robotics/experiment_logs/robotwin_collect_smoke.log"
```

**注意**：`demo_randomized` 可能依赖完整背景纹理，且默认可能尝试采集很多条示范；建议新机首次验收优先使用上述 `demo_preview`，不要一开始就跑大规模采集。`episode_num: 1` 是**目标 1 条成功轨迹，不是只尝试 1 次**。

---

## 10. 创建 XPolicyLab 集成版 DP 环境

此环境是**未来实验室服务器训练、本地 RoboTwin 推理所需的策略端**，不是为 Stanford Push-T 任务准备的。

```bash
conda create -n dp python=3.10 -y
conda activate dp
cd "$HOME/robotics/RoboTwin/XPolicyLab/policy/DP"

# 先查看，不要盲目覆盖版本
cat install.sh

set -o pipefail
bash -e install.sh 2>&1 | tee "$HOME/robotics/experiment_logs/dp_install.log"
```

本次实际 `install.sh` 包含：`torch==2.4.1`、`zarr==2.12.0`、`hydra-core==1.2.0`、`diffusers==0.11.1`、`numba==0.56.4`、`numpy==1.23.5`，还会 `pip install -e .` 安装 DP/XPolicyLab。

### 10.1 必须防范本次实际出现过的 NumPy 2.x 漂移

在本次真实安装中，安装脚本结束后 `numpy` 最终变为 **2.2.6**，而 `numba 0.56.4` 要求 `numpy<1.24`，导入直接失败。另一个依赖带入了 `opencv-python-headless 5.0.0.93`，要求 NumPy ≥2。最终实际成功的修复是：

```bash
conda activate dp

# 先检查，若版本已正确无需重复执行
python -m pip show numpy numba opencv-python-headless
python -m pip check

# 只有检测到 NumPy / OpenCV 版本冲突时才执行下面的钉版本修复：
python -m pip install --force-reinstall --no-deps \
  numpy==1.23.5 opencv-python-headless==4.8.1.78

python -m pip check
```

`--no-deps` 在这里用于**保留已安装的其他依赖不被重新升级**，不是常规安装的默认建议。若 `pip check` 仍指出其他 OpenCV 发行包版本冲突，先用 `python -m pip list | grep -i opencv` 检查，针对错误逐项处理；不要再次升级 NumPy 到 2.x。多个 OpenCV wheel 共享 `cv2` 命名空间，如果发生冲突应清理重复发行包并选择兼容版本，避免长期混装。

### 10.2 验证 DP 核心依赖

```bash
python - <<'PY'
import torch, torchvision, numpy, numba, diffusers, cv2
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('NumPy:', numpy.__version__)
print('Numba:', numba.__version__)
print('Diffusers:', diffusers.__version__)
print('OpenCV:', cv2.__version__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
assert numpy.__version__ == '1.23.5'
PY
python -m pip check
```

本次真实成功结果：`PyTorch 2.4.1+cu121`、`torchvision 0.19.1+cu121`、CUDA 12.1、NumPy 1.23.5、Numba 0.56.4、Diffusers 0.11.1、OpenCV 4.8.1、`pip check` 无冲突。

**到此，`dp` 环境完成的是核心依赖验收，不是 checkpoint 加载和 RoboTwin 闭环推理。** 此后不要再往 `dp` 中安装 Stanford 原版 Push-T 的 Gym/W&B 等依赖。

立即保存成功版本：

```bash
conda env export -n dp > "$HOME/robotics/experiment_logs/dp_clean_environment.yml"
python -m pip freeze > "$HOME/robotics/experiment_logs/dp_clean_pip_freeze.txt"
```

---

## 11. 创建独立 Stanford Push-T 环境

### 11.1 为什么单独创建

你需要完成课程要求的“**运行一次现成预训练模型推理**”。XPolicyLab/DP 仓库在本次检查时没有带可直接用于 RoboTwin 的 DP checkpoint；Stanford 官方公开了 Push-T 预训练权重，但其代码与 XPolicyLab 的 Python 包同名且依赖不同。因此使用第三个环境 `dp-pusht`。

**本次绕过的错误**：在 `dp` 的混合依赖下曾遇到 Gym 0.23 `shared_memory` 参数错位、`reset_async(seed=...)`、新版 W&B 缺少 `generate_id`。新路线直接采用 Stanford 旧版 Gym 0.21 和 W&B 0.13.3，最终运行成功，**无需给官方 Gym/runner 源码逐个打补丁**。

### 11.2 一次性写入实际成功路线对应的 Conda YAML

> 新机器直接从零创建，不要用 `conda create --clone dp`。  
> 这份 YAML **不是 Stanford 原始 `conda_environment.yaml` 的逐字复制**：为 RTX 4060 改用 PyTorch 2.4.1+cu121，移除了 Push-T 单任务推理不需要的旧 PyTorch3D/MuJoCo 等重依赖，增加了本次实际缺失过的 pandas。它是**按用户成功路径整理的最小 Push-T 用途环境**。

以下命令在未激活 Conda 时也可执行（`cat` 由 Shell 执行）：

```bash
cat > "$HOME/robotics/dp_pusht_environment.yml" <<'YAML'
name: dp-pusht
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pip=22.2.2
  - numpy=1.23.3
  - scipy=1.9.1
  - numba=0.56.4
  - h5py=3.7.0
  - diffusers=0.11.1
  - huggingface_hub=0.25.2
  - hydra-core=1.2.0
  - einops=0.4.1
  - dill=0.3.5.1
  - zarr=2.12.0
  - numcodecs=0.10.2
  - accelerate=0.13.2
  - gym=0.21.0
  - pymunk=6.2.1
  # pygame 2.1.2 is installed by pip; conda-forge may not have this exact version
  - shapely=1.8.4
  - scikit-image=0.19.3
  - matplotlib=3.6.1
  - py-opencv=4.6.0
  - av=10.0.0
  - imageio=2.22.0
  - imageio-ffmpeg=0.4.7
  - scikit-video=1.1.11
  - wandb=0.13.3
  - protobuf=3.20.3
  - tqdm=4.64.1
  - click=8.0.4
  - psutil=5.9.2
  - pandas=1.5.3
  - pip:
      - pygame==2.1.2
YAML

mamba env create -f "$HOME/robotics/dp_pusht_environment.yml"
conda activate dp-pusht
python --version
python -m pip check
```

**验收**：Python 3.9.x；`pip check` 无冲突。本次最初 YAML 曾把 `pygame=2.1.2` 写在 Conda 依赖区，Mamba 报 package 不存在；上面已修正为 pip。最初又漏了 `pandas` 和 `pytz`：上面现在使用 Conda 安装 `pandas=1.5.3`，会解析它的依赖；**不要给 pandas 加 `--no-deps`**。

如需重建已有的旧 `dp-pusht`，先确认**仅删除该环境**：`conda deactivate; conda env remove -n dp-pusht -y`。不要删 `RoboTwin`、`dp` 或 checkpoint。

### 11.3 安装适配 RTX 4060 的 CUDA 版 PyTorch

```bash
conda activate dp-pusht
python -m pip install \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import sys, torch, torchvision, gym, numpy, numba, wandb, diffusers, pandas
print('Python:', sys.version.split()[0])
print('Torch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('CUDA:', torch.version.cuda, torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(none)')
print('Gym:', gym.__version__)
print('NumPy:', numpy.__version__)
print('Numba:', numba.__version__)
print('W&B:', wandb.__version__)
print('Pandas:', pandas.__version__)
assert torch.cuda.is_available()
assert gym.__version__ == '0.21.0'
PY
python -m pip check
```

本次成功版本：Python 3.9.18，PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121、Gym 0.21.0、CUDA 可用、RTX 4060 可见、`pip check` 无冲突。

---

## 12. 下载预训练模型并完成一次真实推理

### 12.1 克隆 Stanford 官方仓库，不覆盖 RoboTwin

```bash
conda activate dp-pusht
cd "$HOME/robotics"
git clone https://github.com/real-stanford/diffusion_policy.git diffusion_policy_official
cd diffusion_policy_official

git rev-parse HEAD | tee "$HOME/robotics/experiment_logs/stanford_dp_commit.txt"
```

**避免同名导入冲突**：只在 `dp-pusht` 中使用本仓库代码，进入根目录执行脚本；不在 `dp-pusht` 中安装 XPolicyLab/DP 的可编辑包，也不在 `dp` 中执行这套 Stanford 脚本。

```bash
cd "$HOME/robotics/diffusion_policy_official"
python - <<'PY'
import diffusion_policy
print('Package path:', list(diffusion_policy.__path__))
from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace
from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner
print('Stanford Push-T core imports OK')
PY
```

如果 package path 错误地指向 `~/robotics/RoboTwin/XPolicyLab/...`，先排查 `PYTHONPATH` 和是否在错误的 Conda 环境中。干净的 `dp-pusht` 不应出现此问题。不要通过反复安装两个同名包来解决。

### 12.2 下载本次成功使用的 Stanford Push-T checkpoint

```bash
cd "$HOME/robotics/diffusion_policy_official"
mkdir -p data
wget -c -O data/pusht_pretrained.ckpt \
  'https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt'

ls -lh data/pusht_pretrained.ckpt
```

权重接近 1GB，网络中断后可用 `wget -c` 尝试续传。若检查得到明显很小的 HTML/错误页，不要拿它当 checkpoint；先核对链接和下载日志。

### 12.3 仅评估 1 条轨迹，避免笔记本开太多仿真进程

**本次实际使用了 `eval_pusht_one.py`**。它从官方 `eval.py` 复制，只在读取 checkpoint 内配置后缩小评估次数；不训练模型、不修改模型权重。新系统按下面生成：

```bash
cd "$HOME/robotics/diffusion_policy_official"
cp eval.py eval_pusht_one.py

python - <<'PY'
from pathlib import Path
p = Path('eval_pusht_one.py')
s = p.read_text()
old = "    cfg = payload['cfg']"
new = """    cfg = payload['cfg']
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test = 1
    cfg.task.env_runner.n_test_vis = 1
    cfg.task.env_runner.n_envs = 1"""
if s.count(old) != 1:
    raise SystemExit('Stanford eval.py changed; check source before modifying')
p.write_text(s.replace(old, new, 1))
print('One-episode evaluator generated')
PY
```

**不要恢复/套用旧的 Gym 0.23 或新版 W&B 兼容补丁**。现在使用 Gym 0.21、W&B 0.13.3，官方 `async_vector_env.py` 和 `pusht_keypoints_runner.py` 应保持原始版本。若你是从**曾被修改过的本地源码**继续操作，先记录差异，再用 `git restore` 恢复那两个文件；新克隆仓库无需此步骤。

### 12.4 运行真正的模型推理

```bash
conda activate dp-pusht
cd "$HOME/robotics/diffusion_policy_official"
set -o pipefail
python eval_pusht_one.py \
  --checkpoint data/pusht_pretrained.ckpt \
  --output_dir data/pusht_eval_verified \
  --device cuda:0 \
  2>&1 | tee "$HOME/robotics/experiment_logs/dp_pusht_verified.log"
```

**验收标准**：程序正常结束，并生成 `eval_log.json` 和至少一个 `.mp4`。首次运行若输出目录存在，会询问 `Overwrite? [y/N]`，确认是自己之前的测试目录后再输入 `y`，否则换新的 `--output_dir`。不会自动显示 RoboTwin 式的实时 GUI；官方 Push-T 基准通常输出视频文件。

```bash
cd "$HOME/robotics/diffusion_policy_official"
cat data/pusht_eval_verified/eval_log.json
find data/pusht_eval_verified -type f -name '*.mp4'

# 播放查找到的第一段视频
ffplay -autoexit "$(find data/pusht_eval_verified -type f -name '*.mp4' -print -quit)"
```

本次真实用户机器的验收记录：

```json
{
  "test/mean_score": 0.9953731295186483,
  "test/sim_max_reward_4300000": 0.9953731295186483,
  "test/sim_video_4300000": "data/pusht_eval_verified/media/quf6yoon.mp4"
}
```

**解释**：0.9954 是这次单轨迹评估得分，**不能直接说成 99.54% 的统计成功率**。另一台机器的 seed、视频名、分数不保证相同。

**至此课程所要求的“运行一次现成推理”完成；但这不是 RoboTwin 上的 DP 联合推理**，Push-T 权重不能直接用于 Aloha 或 Piper 的关节动作任务。

---

## 13. 保存日志、版本和可复现证据

```bash
mkdir -p "$HOME/robotics/experiment_logs"

# 系统基础信息
{
  date -Is
  cat /etc/os-release
  uname -a
  nvidia-smi
  nvcc --version
  vulkaninfo --summary
} > "$HOME/robotics/experiment_logs/system_versions.txt" 2>&1

# 三个环境的完整 Conda 导出与 pip 版本
for e in RoboTwin dp dp-pusht; do
  conda env export -n "$e" > "$HOME/robotics/experiment_logs/${e}_full.yml"
  "$HOME/miniforge3/envs/$e/bin/python" -m pip freeze \
    > "$HOME/robotics/experiment_logs/${e}_pip_freeze.txt"
done

# 两个仓库版本与本地更改
cd "$HOME/robotics/RoboTwin"
git rev-parse HEAD > "$HOME/robotics/experiment_logs/RoboTwin_git_sha.txt"
git submodule status > "$HOME/robotics/experiment_logs/XPolicyLab_submodule_sha.txt"
git diff > "$HOME/robotics/experiment_logs/RoboTwin_local_changes.patch"

cd "$HOME/robotics/diffusion_policy_official"
git rev-parse HEAD > "$HOME/robotics/experiment_logs/Stanford_DP_git_sha.txt"
git status --short > "$HOME/robotics/experiment_logs/Stanford_DP_git_status.txt"
git diff > "$HOME/robotics/experiment_logs/Stanford_DP_local_changes.patch"
```

注意 `git diff` 不包含未跟踪文件（如 `eval_pusht_one.py`）；还需**额外保存该脚本及 YAML 文件**。推荐把以下文件一起纳入实验提交或归档：

- `~/robotics/dp_pusht_environment.yml`
- `~/robotics/diffusion_policy_official/eval_pusht_one.py`
- `~/robotics/experiment_logs/dp_pusht_verified.log`
- `~/robotics/diffusion_policy_official/data/pusht_eval_verified/eval_log.json`
- `~/robotics/diffusion_policy_official/data/pusht_eval_verified/media/*.mp4`
- `~/robotics/experiment_logs/robotwin_preview.log`
- `~/robotics/RoboTwin/data/demo_preview/beat_block_hammer/aloha_agilex/video/*.mp4`

`conda env export` 可能包含本机的 `prefix:`；在另一机器重建前删除/调整这项。**安全提醒**：不要把 `.env` 密钥、Hugging Face 访问令牌、W&B 登录凭据或包含私密路径的敏感日志提交到公开仓库。

---

## 14. 服务器训练与本地 RoboTwin DP 联合推理（后续可选）

### 14.1 实际计划（目前不是已通过的验收项）

```text
实验室服务器
  └── 下载 RoboTwin 专家 HDF5 → 转换 Zarr → XPolicyLab/DP 训练
      └── 输出匹配机器人/任务/动作定义的 checkpoint
              ↓ 复制 checkpoint 和完整配置
本地 Ubuntu
  ├── (RoboTwin)：SAPIEN 场景、机械臂、相机、GUI/视频
  └── (dp)：加载 checkpoint、预测动作、与仿真侧联机
```

已经完成的 Push-T 推理只是单独验证 Stanford 预训练模型；它不能替代这条链路。

### 14.2 原指南中的数据与训练入口，留作未来参考

这些命令来自最初指南和对应的 XPolicyLab 脚本设计，**本次没有实际跑通服务器训练，不应在新系统的“基础环境验收”阶段强制执行**。未来上服务器前先检查所在版本 README、`process_data.sh`、`train.sh`、`deploy.yml`。

```bash
# 服务器上下载示范数据：例子为 beat_block_hammer
conda activate RoboTwin
cd "$HOME/robotics/RoboTwin"
bash scripts/download_xpolicylab_data.sh beat_block_hammer

# 必须检查数据实际目录与 HDF5 schema
find data/demo_clean/beat_block_hammer -name '*.hdf5' | head
```

原指南的 DP 数据转换示例：

```bash
# 只有在确认数据路径及 robot/action schema 与脚本相符后才执行
conda activate dp
cd "$HOME/robotics/RoboTwin/XPolicyLab/policy/DP"
bash process_data.sh RoboTwin beat_block_hammer arx_x5 joint 20
```

原指南的训练示例：

```bash
# 服务器上、有效 Zarr 数据和合适显存/参数准备完成后才启动
conda activate dp
cd "$HOME/robotics/RoboTwin/XPolicyLab/policy/DP"
bash train.sh RoboTwin beat_block_hammer arx_x5 joint 0 0
```

原指南的评估入口是 `scripts/eval_policy.sh`，需要训练好的**匹配 checkpoint**，并配置 `--policy-conda-env dp`、`--eval-env-conda-env RoboTwin`。**没有有效权重，不要将 dry-run 或专家演示认作 RoboTwin DP 推理完成**。

### 14.3 Aloha 与 Piper 是可切换的机器人选择

- 已下载的 `assets/embodiments/` 包含 Aloha AgileX 与 Piper；**无需重新安装 RoboTwin，也不必重新下载整套资产**。
- 切换到 Piper 时，需要选择相应的 `embodiment`，确认该机器人在本版本的任务与 CuRobo YAML 配置齐全。某些旧镜像版需给 Piper 补左右臂规划 YAML，**你的官方版本不要未经检查就照搬镜像版补丁**。
- 不同机器人观测、关节结构、动作空间可能不同；**DP 权重不能简单跨机器人复用**，训练数据和 deployment 配置必须匹配。

---

---
## 15. 最终验收清单与文件位置

### 15.1 新系统最小验收

- [ ] Ubuntu 22.04、NVIDIA 驱动、Vulkan 都正常；`nvidia-smi` 与 `vulkaninfo` 可见独显。
- [ ] 系统级 CUDA 12.1 编译器 `nvcc --version` 正常。
- [ ] Miniforge、Conda、Mamba 正常；三个环境分开存在。
- [ ] `RoboTwin` 的 PyTorch/CUDA 与 SAPIEN、MPLib、CuRobo 导入通过。
- [ ] `assets/embodiments`、`assets/objects`、`assets/background_texture/seen` 均存在且经过 ZIP 完整性验证。
- [ ] `beat_block_hammer` 的 `demo_preview` 至少一条专家示范成功并产生 MP4。
- [ ] `dp` 的 PyTorch、NumPy 1.23.5、Numba 0.56.4、Diffusers、OpenCV 导入通过，`pip check` 无冲突。
- [ ] `dp-pusht` 的 Python 3.9、PyTorch cu121、Gym 0.21、Stanford 推理代码导入通过。
- [ ] 官方 Push-T checkpoint 完整，`eval_pusht_one.py` 产生 `eval_log.json` 和 MP4。
- [ ] 保存三个环境的导出文件、两个 Git SHA、终端日志和视频。

### 15.2 以后找不到文件时看这里

| 内容 | 路径 |
|---|---|
| RoboTwin 项目 | `~/robotics/RoboTwin/` |
| RoboTwin 配置 | `~/robotics/RoboTwin/env_cfg/task_config/` |
| RoboTwin 机器人/物体/纹理 | `~/robotics/RoboTwin/assets/` |
| RoboTwin 专家示范视频 | `~/robotics/RoboTwin/data/demo_preview/beat_block_hammer/aloha_agilex/video/` |
| XPolicyLab 集成 DP | `~/robotics/RoboTwin/XPolicyLab/policy/DP/` |
| Stanford 官方 DP | `~/robotics/diffusion_policy_official/` |
| Push-T checkpoint | `~/robotics/diffusion_policy_official/data/pusht_pretrained.ckpt` |
| Push-T 推理结果 | `~/robotics/diffusion_policy_official/data/pusht_eval_verified/` |
| 实验日志与 Conda 导出 | `~/robotics/experiment_logs/` |
| Conda 环境 | `~/miniforge3/envs/{RoboTwin,dp,dp-pusht}/` |
| CUDA Toolkit | `/usr/local/cuda-12.1/` |

### 15.3 参考源码/来源

- [RoboTwin 官方仓库](https://github.com/RoboTwin-Platform/RoboTwin)
- [RoboTwin 官方安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
- [XPolicyLab / DP 适配器](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/DP)
- [Stanford Diffusion Policy 官方仓库](https://github.com/real-stanford/diffusion_policy)
- [Stanford 原始 Conda 配置](https://github.com/real-stanford/diffusion_policy/blob/main/conda_environment.yaml)
- [PyTorch 官方历史安装版本](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA Toolkit 安装说明](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Hugging Face RoboTwin 2.0 资源](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)

---
