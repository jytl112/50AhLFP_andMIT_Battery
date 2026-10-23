# 50AhLFP_andMIT_Battery
**Physics-Informed Neural Network and Computational Fluid Dynamics Coupled Heat Transfer for Battery State of Health.**
## English
### Overview
This repository contains training and evaluation code for battery state-of-health (SOH) estimation using the MIT battery dataset and experimental data from a 50 Ah lithium iron phosphate (LiFePO₄, LFP) cell. It supports physics-informed neural networks (PINNs), transfer learning, comparisons with data-driven models, and chronological multi-step forecasting.
The project provides four experimental workflows:
|Workflow|Purpose|Models|
|---|---|---|
|MIT benchmark|Evaluate SOH tracking across three MIT batches with fixed battery-level splits and three random seeds.|PINN, MLP_NoPhysics, LSTM, CNN, Transformer|
|Source pretraining|Train source models on MIT Batch 1 for subsequent adaptation to the 50 Ah cell.|PINN, LSTM, CNN, Transformer|
|Sparse target adaptation|Adapt source models using a small number of labeled 50 Ah cycles and evaluate the remaining trajectory.|PINN, LSTM, CNN, Transformer|
|Multi-step forecasting|Train using an initial cycle interval, freeze the models, and predict SOH 1, 10, and 50 cycles ahead.|PINN, LSTM, CNN, Transformer|
### Repository structure
```text
.
├── README.md
├── requirements.txt
├── run.py                         # Unified training and prediction entry point
├── code/
│   ├── mit/                       # MIT benchmark training
│   ├── source/                    # Source models and pretraining
│   ├── trajectory/                # Sparse target adaptation and evaluation
│   ├── multistep/                 # Prefix training, forecasting, and scoring
│   └── predict_mit.py             # Replay saved MIT predictions
├── config/
│   ├── data_paths.json            # External data and checkpoint locations
│   ├── mit_protocol.json          # MIT splits and training settings
│   └── forecast_protocol.json     # Multi-step evaluation protocol
├── SOURCE_MANIFEST.json           # Code provenance and file hashes
└── VERIFICATION.json              # Recorded reproduction checks
```

### Requirements

The recorded execution environment uses Python 3.13.5, TensorFlow 2.20.0rc0, and Keras 3.11.1. Dependency versions are listed in [requirements.txt](requirements.txt).

Install them in your Python environment:

```bash
python -m pip install -r requirements.txt
```

### Data and checkpoints

**The code package does not include datasets, pretrained weights, or archived prediction results.** Training requires the prepared input files; replaying saved predictions also requires the corresponding checkpoints and selection records.

Edit [config/data_paths.json](config/data_paths.json) to match your local files. The existing entries are local directory examples and may need updating after the repository is moved. An example configuration is:

```json
{
  "mit": "D:/battery_data/MIT_Pre/01_benchmark",
  "source": "D:/battery_data/50LFP_Pre/01_trajectory/inputs/MIT_source",
  "trajectory": "D:/battery_data/50LFP_Pre/01_trajectory",
  "multistep": "D:/battery_data/50LFP_Pre/02_multistep"
}
```

Replace these paths with the locations of your prepared experiment files. Relative paths are resolved from the repository root.

|Configuration key|Required external content|
|---|---|
|`mit`|Prepared `inputs/MIT.csv`. Prediction replay additionally uses `selected/*_configs.json`, selected model weights, and archived prediction CSVs under `results/`.|
|`source`|`source_inputs.npz` and `source_scale.json` for source pretraining; source-model configuration files and weights for target adaptation.|
|`trajectory`|`inputs/50Ah_target.xlsx`. Prediction replay additionally requires selected target weights, validation-selection records, and training protocols under `results/`.|
|`multistep`|`inputs/operating_inputs.csv`, `prefix_operating_inputs.csv`, `six_prefix_labels.csv`, and `heldout_raw_targets.csv`. Prediction replay additionally requires the selected model weights and refitting records.|

The code starts from prepared, cycle-level inputs. It does not download raw datasets or reproduce the earlier raw-data extraction notebooks. The `check` command reports missing files before a run begins.

### Quick start

Run the following commands from the repository root after configuring the external paths.

**Check inputs and code syntax:**

```bash
python run.py check
python run.py check --check-action train
python run.py check --task source --check-action train
```

**Replay saved predictions:**

```bash
python run.py predict --task all
python run.py predict --task mit --model PINN
python run.py predict --task trajectory
python run.py predict --task multistep
```

Prediction mode loads saved weights and evaluates their outputs; it does not retrain the models. The MIT replay also checks consistency with archived predictions.

**Train models:**

```bash
# MIT benchmark: select PINN or all five models
python run.py train --task mit --model PINN
python run.py train --task mit --model all

# MIT source pretraining for the 50 Ah transfer experiment
python run.py train --task source --model PINN
python run.py train --task source --model all

# Four-model target adaptation or chronological forecasting
python run.py train --task trajectory
python run.py train --task multistep
```

The target workflows use the source checkpoints specified by the `source` configuration entry. To use newly pretrained models, point that entry to the new run's `source/inputs/MIT_source` directory.

Each run creates a separate output directory under `runs/`. Use `--output NEW_DIRECTORY` to choose another location or `--paths PATHS.json` to supply a different path configuration. Existing external inputs and archived weights are preserved.

### Experimental protocols

|Experiment|Protocol and interpretation|
|---|---|
|MIT benchmark|Batch 1/2 use 38 training, 1 validation, and 1 test cell; Batch 3 uses 36/1/1. Seeds: 42, 2026, and 3407. Training labels use a trailing five-record median and cumulative minimum; evaluation uses raw SOH. The inputs include lagged capacity loss, so this is capacity-conditioned SOH tracking.|
|Sparse 50 Ah adaptation|Cycles 1, 104, 207, 310, 412, and 515 provide training labels; cycles 618 and 721 provide validation labels. The other 713 records are evaluated. Because late-cycle labels select the model, this is retrospective trajectory estimation with sparse calibration.|
|Multi-step forecasting|Model selection and refitting use labels through cycle 515. Frozen models are evaluated at 157 common origins, cycles 515–671, for horizons of 1/10/50 cycles. Future cumulative operating features are held at their origin values while the cycle number advances.|

The MIT benchmark PINN and the 50 Ah transfer PINN use separate implementations: an 8 × 64 backbone for the benchmark and a 12 × 96 backbone for transfer. Their objectives and input definitions should be interpreted within their respective workflows.

The 50 Ah experiments use one physical target cell. Multiple cycles, overlapping forecast origins, or repeated random seeds do not establish performance across independent target cells. Results remain conditional on the recorded data splits, preprocessing, and future-input assumptions.

### Reproduction status

The recorded packaging checks reloaded 45 MIT benchmark models and eight 50 Ah target models, reproduced their predictions, and checked required inputs and code syntax. Full retraining was not repeated during packaging. See [VERIFICATION.json](VERIFICATION.json) for the recorded scope and numerical differences, and [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) for code provenance.

## 简体中文

### 项目简介

本仓库提供基于 MIT 电池数据集与一块 50 Ah 磷酸铁锂电池实验数据的健康状态（SOH）估计与预测代码，包含物理信息神经网络（PINN）、迁移学习、数据驱动模型对比和时间顺序多步预测。

主要功能包括：

- **MIT 基准训练与评价**：对三个批次采用固定电池级划分和三个随机种子，比较 PINN、LSTM、CNN 和 Transformer。
- **源域预训练**：在 MIT Batch 1 上训练用于 50 Ah 迁移实验的源模型。
- **稀疏标签迁移**：使用少量目标电池标签微调，评价其余循环的原始 SOH。
- **多步预测**：在前缀区间完成选模与训练，冻结权重后预测提前 1、10、50 个循环的 SOH。

### 文件与数据

`code/mit`、`code/source`、`code/trajectory` 和 `code/multistep` 分别对应上述四个流程；`run.py` 为统一入口；`config` 保存划分、评估协议和外部数据位置。

**本仓库的代码包不包含数据集、预训练权重和历史预测结果。** 训练需要已经处理好的输入文件，复算已有预测还需要相应的模型权重和选模记录。代码从固定的循环级特征开始，不自动下载原始数据或重做历史原始数据提取。

使用前请修改 [config/data_paths.json](config/data_paths.json) 中的四个路径：

- `mit`：MIT 基准数据、所选模型配置和结果目录。
- `source`：完整的源域训练数组、缩放信息、源模型配置和权重目录。
- `trajectory`：50 Ah 稀疏标定实验目录。
- `multistep`：50 Ah 多步预测实验目录。

路径可以是绝对路径，也可以相对于仓库根目录。上方英文说明列出了目录内容和配置示例。移动仓库后应重新核对路径；`check` 会列出缺失文件。

### 安装与运行

记录的运行环境为 Python 3.13.5、TensorFlow 2.20.0rc0 和 Keras 3.11.1。安装依赖并配置好数据路径后，在仓库根目录运行：

```bash
python -m pip install -r requirements.txt

# 检查预测或训练所需文件
python run.py check
python run.py check --check-action train
python run.py check --task source --check-action train

# 加载已保存权重并复算预测，不启动训练
python run.py predict --task all

# MIT 基准训练
python run.py train --task mit --model PINN
python run.py train --task mit --model all

# 50 Ah 迁移实验的源域预训练
python run.py train --task source --model PINN
python run.py train --task source --model all

# 目标域微调和多步预测训练
python run.py train --task trajectory
python run.py train --task multistep
```

MIT 的 `all` 包含五个模型；源域预训练的 `all` 包含 PINN、LSTM、CNN、Transformer。两种目标训练流程分别执行四个模型。

目标训练默认使用 `source` 指向的源域检查点。如果使用刚重新训练的源模型，请将该路径改为新运行目录中的 `source/inputs/MIT_source`。

每次执行都会在新的 `runs/` 子目录中产生结果，外部原数据与归档权重保持不变。可用 `--output 新目录` 指定输出位置，用 `--paths 路径配置.json` 指定另一份数据位置配置。后续打包时应排除运行产生的 `runs/`。

实验口径

|实验|划分与解释|
|---|---|
|MIT 基准|Batch 1/2 为 38/1/1 个训练、验证、测试电池，Batch 3 为 36/1/1；种子为 42、2026、3407。训练标签采用向后五记录中值和累计最小值，评价采用原始 SOH。输入含前一周期容量损失，属于具有历史容量观测的 SOH 跟踪。|
|50 Ah 稀疏标定|循环 1/104/207/310/412/515 用于微调，618/721 用于验证，其余 713 条记录评价。后期标签参与选模，因此该实验属于稀疏标定条件下的回顾性轨迹估计。|
|50 Ah 多步预测|使用截至循环 515 的标签选模并重拟合，随后冻结权重；在 515—671 的 157 个共同起点预测提前 1/10/50 循环。未来五项运行累积均值保持起点值，循环号递增。|

MIT 基准与 50 Ah 迁移采用各自的网络实现，主干分别为 8 × 64 和 12 × 96；输入和损失函数应结合对应流程解释。50 Ah 实验来自一块物理电池，多循环记录、重叠预测起点和重复随机种子不等同于独立电池验证。

已有核验

代码整理时已重新加载 45 组 MIT 基准模型和 8 组 50 Ah 目标模型，完成预测复算、必要输入和代码语法检查；整理过程未重新执行全量训练。核验范围与误差见 [VERIFICATION.json](VERIFICATION.json)，代码来源和哈希见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。
