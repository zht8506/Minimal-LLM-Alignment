
\[ [English](README.md) | 中文\]

使用最少的代码行在PyTorch中实现主流的llm 对齐算法！

从繁杂的框架中脱离，学习对齐算法的核心。

## 🔥 最新

- ```[2025/9]``` 支持SFT和DPO，代码简洁。

- ```[2025/8]``` **Minimal-LLM-Alignment**开源.

## 📊 开始

### 安装依赖
```bash
conda create --name myenv python=3.10
pip install -r requirements.txt
```
### 训练

#### SFT训练
```bash
python train.py example/qwen2.5-sft.yml
```

#### DPO训练
```bash
python train.py example/qwen2.5-dpo.yml
```
#### 命令行覆盖

```bash
python train.py config.yml --overrides lr=2e-5 batch_size=32
```

## 📁 数据构建方式

### 使用huggingface数据

为了快速上手我们的项目，我们支持三个 huggingface 数据集，可以用来进行DPO和SFT训练。这些数据集包括： ```Anthropic/hh-rlhf```([link](https://huggingface.co/datasets/Anthropic/hh-rlhf))、```stanfordnlp/SHP```([link](https://huggingface.co/datasets/stanfordnlp/SHP))、```HuggingFaceH4/stack-exchange-preferences```([link](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)).

此外，我们还支持自己构建数据集进行对齐训练。


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


## 🎯 领先的 LLM 对齐方法
注意：部分算法缺乏官方实现，因此我采用了优秀的开源版本。

### Offline RL
| **Release** | **Method** | **Reference** | **Notes** | **Link** |
| --- | --- | --- | --- | --- |
| 2023/05 | **DPO** | Direct preference optimization: Your language model is secretly a reward model | NeurIPS 2023 | [paper](https://arxiv.org/abs/2305.18290)/[code](https://github.com/eric-mitchell/direct-preference-optimization)|
| 2023/10 | **IPO** | A General Theoretical Paradigm to Understand Learning from Human Preferences | AISTATS 2024 | [paper](https://arxiv.org/abs/2310.12036)|
| 2024/02 | **KTO** | KTO: Model Alignment as Prospect Theoretic Optimization | ICML 2024 | [paper](https://arxiv.org/abs/2402.01306)/[code](https://github.com/ContextualAI/HALOs)|
| 2024/03 | **ORPO** | Orpo: Monolithic preference optimization without reference model | EMNLP 2024 | [paper](https://arxiv.org/abs/2403.07691)/[code](https://github.com/xfactlab/orpo)|
| 2024/05 | **SimPO** | Simpo: Simple preference optimization with a reference-free reward | NeurIPS 2024 | [paper](https://arxiv.org/abs/2405.14734)/[code](https://github.com/princeton-nlp/SimPO)|

### Online RL
| **Release** | **Method** | **Reference** | **Notes** | **Link** |
| --- | --- | --- | --- | --- |
| 2017/07 | **PPO** | Proximal Policy Optimization Algorithms | Arxiv | [paper](https://arxiv.org/abs/1707.06347)/[code](https://github.com/nikhilbarhate99/PPO-PyTorch) |
| 2017/07 | **GRPO** | Deepseekmath: Pushing the limits of mathematical reasoning in open language models | Arxiv | [paper](https://arxiv.org/abs/2402.03300)/[code](https://github.com/lsdefine/simple_GRPO) |





