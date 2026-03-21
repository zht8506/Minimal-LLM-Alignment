# Easy-LLM-Post-Training

\[English | [中文](README_zh.md)\]

Implement popular LLM post-training algorithms in PyTorch with easy code!

以简洁代码，基于 PyTorch 实现主流大模型后训练算法！

Break free from complex frameworks and focus on the core logic of LLM post training algorithms.

摆脱繁杂框架束缚，专注于大模型后训练算法的核心逻辑。


## 🚀 Supported Algorithms

- Supervised Fine-Tuning (SFT)
- Direct Fine-Tuning (DFT) [[Paper](https://arxiv.org/abs/2508.05629)]
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)
- More coming soon

## 📊 Getting Start

### Env

```bash
conda create --name myenv python=3.10
pip install -r requirements.txt
```

### Data

You can get started quickly using the sample data in the ```data```, including SFT data, DPO data, etc.

### Training

Taking SFT as an example:

```bash
cd sft/
bash train.sh
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


