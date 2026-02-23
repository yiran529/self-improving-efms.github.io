# Pointmass PyTorch Pipeline

本仓库已把原 notebook 流程整理为可命令行运行的工程化结构，支持单机单设备（CPU 或单卡 GPU）。

## 目录结构

```text
.
├─ config/
│  └─ default.json
├─ src/
│  └─ pointmass/
│     ├─ config.py
│     ├─ env.py
│     ├─ model.py
│     ├─ trainer.py
│     └─ utils.py
├─ scripts/
│  ├─ run_generate_data.py
│  ├─ run_train_sft.py
│  ├─ run_train_self_improve.py
│  ├─ run_evaluate.py
│  └─ run_all.py
└─ pointmass_notebook.py
```

## 安装依赖

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

## 启动实验（CLI）

下面给出标准流程。

### 1) 生成数据（PD 控制器）

```bash
python pointmass_notebook.py --config config/default.json generate-data
```

快速小样本调试：

```bash
python pointmass_notebook.py --config config/default.json generate-data --num-episodes 100
```

### 2) Stage 1: 监督训练（SFT）

```bash
python pointmass_notebook.py --config config/default.json train-sft
```

可覆盖训练步数：

```bash
python pointmass_notebook.py --config config/default.json train-sft --num-updates 200
```

### 3) Stage 2: 自改进训练（REINFORCE）

```bash
python pointmass_notebook.py --config config/default.json train-self-improve
```

可覆盖迭代数：

```bash
python pointmass_notebook.py --config config/default.json train-self-improve --num-iterations 50
```

### 4) 评估

```bash
python pointmass_notebook.py --config config/default.json evaluate
```

使用确定性动作（均值动作）：

```bash
python pointmass_notebook.py --config config/default.json evaluate --deterministic-action
```

### 5) 轨迹视频可视化（仅生成视频，不弹窗）

数据轨迹视频：

```bash
python pointmass_notebook.py --config config/default.json visualize-dataset
```

Stage1 策略轨迹视频：

```bash
python pointmass_notebook.py --config config/default.json visualize-stage1
```

Stage2 策略轨迹视频：

```bash
python pointmass_notebook.py --config config/default.json visualize-stage2
```

## scripts 入口（可选）

与上面命令等价：

```bash
python scripts/run_generate_data.py --config config/default.json
python scripts/run_train_sft.py --config config/default.json
python scripts/run_train_self_improve.py --config config/default.json
python scripts/run_evaluate.py --config config/default.json
python scripts/run_visualizations.py --config config/default.json
```

## 一键全流程

按默认配置串行执行：数据生成 -> 数据可视化 -> Stage1 -> Stage1可视化 -> Stage2 -> Stage2可视化 -> 评估。

```bash
python scripts/run_all.py --config config/default.json
```

快速调试（小规模）：

```bash
python scripts/run_all.py --config config/default.json --num-episodes 100 --sft-updates 200 --self-improve-iters 20 --eval-episodes 10
```

可覆盖视频参数：

```bash
python scripts/run_all.py --config config/default.json --vis-episodes 5 --vis-fps 10 --vis-max-steps 140
```

跳过某些阶段：

```bash
python scripts/run_all.py --config config/default.json --skip-generate --skip-sft
```

## 产物位置

- 数据：`artifacts/data/pointmass_dataset.pkl`
- Stage1 checkpoint：`artifacts/checkpoints/sft_last.pt`
- Stage2 checkpoint：`artifacts/checkpoints/self_improve_last.pt`
- Dataset 视频：`artifacts/videos/dataset_trajectories.mp4`
- Stage1 视频：`artifacts/videos/stage1_trajectories.mp4`
- Stage2 视频：`artifacts/videos/stage2_trajectories.mp4`

若当前环境的 `ffmpeg` 不可用，程序会自动回退并生成同名 `.gif` 动图。

可在 `config/default.json` 中统一修改路径和超参数。
