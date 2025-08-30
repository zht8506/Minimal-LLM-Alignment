# Minimal-LLM-Alignment

\[ English | [中文](README_zh.md) \]

Implement popular llm alignment algorithms in PyTorch with minimal lines of code!


## 🔥 News

- ```[2025/9]``` Supports SFT and DPO with easy codes.

- ```[2025/8]``` The **Minimal-LLM-Alignment** repository was open-sourced.

## 🚀 Supported Methods

| Method               |     Type    |    Full   |       LoRA         |
| ---------------------- | ------------------ | ------------------ | ------------------ |
| SFT           | SFT | :white_check_mark: | In Development | 
| DPO           | offline RL | :white_check_mark: | In Development |


## 📊 Dataset

### Huggingface Dataset

To help you quickly get started with our project, we support three hugging face datasets for both DPO and SFT. These include:
-hh:
-jj
-ss

### Data Preparation Guide
#### 1. SFT Dataset Format

SFT datasets use conversation format, where each sample contains a conversation sequence:

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

#### 2. DPO Dataset Format

DPO datasets contain preference comparison information, where each sample includes:
- `conversations`: Conversation context
- `chosen`: Preferred response
- `rejected`: Non-preferred response

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


## 🎯 Awesome LLM Alignment Methods
Note: Some algorithms lack official implementations; hence, I adopt the excellent open-source version.

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


