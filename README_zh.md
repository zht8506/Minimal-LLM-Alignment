
\[ [English](README.md) | 中文\]

一个轻量级的大语言模型对齐训练框架，支持SFT（监督微调）和DPO（直接偏好优化）等主流对齐方法。

## 🛠️ 安装要求

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+
- CUDA 11.8+ (GPU训练)

### 安装依赖
```bash
pip install torch transformers datasets accelerate wandb omegaconf
```

## 📊 自定义数据构建

### 1. SFT数据集格式

SFT数据集使用对话格式，每个样本包含一个对话序列：

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Describe a process of making crepes."
      },
      {
        "role": "assistant",
        "content": "Making crepes is an easy and delicious process! Here are step-by-step instructions..."
      }
    ]
  }
]
```

### 2. DPO数据集格式

DPO数据集包含偏好对比信息，每个样本包含：
- `conversations`: 对话上下文
- `chosen`: 偏好回答
- `rejected`: 非偏好回答

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "How can I best prepare for a job interview?"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "Preparing for a job interview requires a combination of research, practice, and self-reflection..."
    },
    "rejected": {
      "from": "gpt",
      "value": "Here are some tips to help you prepare for a job interview..."
    }
  }
]
```

### 3. 数据转换工具

## 🚀 使用方法

### 1. 基础训练

#### SFT训练
```bash
python train.py example/pythia28-sft.yml
```

#### DPO训练
```bash
python train.py example/pythia28-dpo.yml
```

### 2. 自定义配置

创建自定义配置文件：

```yaml
# 基础配置
seed: 0
exp_name: my_experiment
datasets: 
  - type: custom_dataset
    path: path/to/your/dataset.json

# 训练参数
trainer: BasicTrainer
optimizer: AdamW
lr: 1e-5
total_batch_size: 16
gradient_accumulation_steps: 4
max_length: 512
n_epochs: 100

# 模型配置
model:
  name_or_path: your/model/path
  policy_dtype: float16

# 损失函数配置
loss:
  name: dpo  # 或 sft
  beta: 0.1  # DPO参数
```

### 3. 命令行参数覆盖

```bash
python train.py config.yml --overrides lr=2e-5 batch_size=32
```

### 4. 分布式训练

使用FSDP训练器进行分布式训练：

```bash
python train.py config.yml --overrides trainer=FSDPTrainer
```

## 📁 配置文件详解

### 主要配置项

- **基础配置**: 实验名称、随机种子、调试模式
- **数据集配置**: 数据集类型、路径、加载方式
- **训练配置**: 学习率、批次大小、优化器、训练步数
- **模型配置**: 模型路径、数据类型、检查点加载
- **损失配置**: 损失函数类型、参数设置
- **输出配置**: 输出目录、日志记录、模型保存

### 环境变量

- `WANDB_CACHE_DIR`: WandB缓存目录
- `XDG_CACHE_HOME`: 临时文件缓存目录

## 🔧 高级功能

### 1. 模型检查点加载

```yaml
model:
  archive: path/to/checkpoint.pt  # 加载预训练权重
```

### 2. 梯度累积

```yaml
gradient_accumulation_steps: 4  # 梯度累积步数
```

### 3. 激活检查点

```yaml
activation_checkpointing: true  # 启用激活检查点以节省内存
```

### 4. 混合精度训练

```yaml
model:
  policy_dtype: bfloat16  # 使用混合精度训练
```

## 📈 监控和评估

### 1. WandB集成

```yaml
wandb:
  enabled: true
  entity: your_entity
  project: "Minimal-LLM-Alignment"
```

### 2. TensorBoard支持

```yaml
report_to_tensorboard: true
```

### 3. 评估配置

```yaml
do_first_eval: false
eval_every_steps: 100
n_eval_examples: 256
```
