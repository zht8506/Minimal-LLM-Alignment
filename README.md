# Easy-LLM-Post-Training

\[English | [中文](README_zh.md)\]

Implement popular LLM post-training algorithms in PyTorch with easy code!

以简洁代码，基于 PyTorch 实现主流大模型后训练算法！

Break free from complex frameworks and focus on the core logic of LLM post training algorithms.

摆脱繁杂框架束缚，专注于大模型后训练算法的核心逻辑。


## 🚀 Supported Algorithms

**SFT-Series**
- Supervised Fine-Tuning (**SFT**) [[Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/sft)]
- Dynamic Fine-Tuning (**DFT**) [[Paper](https://arxiv.org/abs/2508.05629) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/dft)]

**DPO-Series**

- Direct Preference Optimization (**DPO**) [[Paper](https://arxiv.org/abs/2305.18290)]
- Simple Preference Optimization (**SimPO**) [[Paper](https://arxiv.org/abs/2405.14734) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/simpo)]

**PPO-Series**
- Proximal Policy Optimization (**PPO**) [[Paper](https://arxiv.org/abs/1707.06347) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/ppo)]
- Group Relative Policy Optimization (**GRPO**) [[Paper](https://arxiv.org/abs/2402.03300) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/grpo)]
- Group Sequence Policy Optimization (**GSPO**) [[Paper](https://arxiv.org/abs/2507.18071) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/gspo)]
- **REINFORCE++** [[Paper](https://arxiv.org/abs/2501.03262) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/reinforce++)]


## 🔄 Upcoming Methods
- Group Relative Policy Optimization Done Right (**Dr.GRPO**)
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
