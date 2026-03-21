
\[ [English](README.md) | 中文\]

使用最少的代码行在PyTorch中实现主流的llm 对齐算法！

从繁杂的框架中脱离，学习对齐算法的核心。

## 🚀 支持的算法

- Supervised Fine-Tuning (SFT)
- Dynamic Fine-Tuning (DFT)
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)
- 敬请期待

## 📊 开始

### 环境

```bash
conda create --name myenv python=3.10
pip install -r requirements.txt
```

### 数据

您可以利用```data```中的示例数据快速开始，包括SFT数据、DPO数据等。

### Training

以SFT为例：

```bash
cd sft/
bash train.sh
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





