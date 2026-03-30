
\[ [English](README.md) | 中文\]

以简洁代码，基于 PyTorch 实现主流大模型后训练算法！

摆脱繁杂框架束缚，专注于大模型后训练算法的核心逻辑。

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


## 🚀 支持的算法

**SFT-系列**
- Supervised Fine-Tuning (**SFT**) [[Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/sft)]
- Dynamic Fine-Tuning (**DFT**) [[Paper](https://arxiv.org/abs/2508.05629) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/dft)]

**DPO-系列**

- Direct Preference Optimization (**DPO**) [[Paper](https://arxiv.org/abs/2305.18290)]
- Simple Preference Optimization (**SimPO**) [[Paper](https://arxiv.org/abs/2405.14734) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/simpo)]

**PPO-系列**
- Proximal Policy Optimization (**PPO**) [[Paper](https://arxiv.org/abs/1707.06347) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/ppo)]
- Group Relative Policy Optimization (**GRPO**) [[Paper](https://arxiv.org/abs/2402.03300) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/grpo)]
- Group Sequence Policy Optimization (**GSPO**) [[Paper](https://arxiv.org/abs/2507.18071) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/gspo)]
- **REINFORCE++** [[Paper](https://arxiv.org/abs/2501.03262) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/reinforce++)]
- Group Relative Policy Optimization Done Right (**Dr.GRPO**) [[Paper](https://arxiv.org/abs/2503.20783) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/dr.grpo)]
- Odds Ratio Preference Optimization (**ORPO**) [[Paper](https://arxiv.org/abs/2403.07691) | [Code](https://github.com/zht8506/Easy-LLM-Post-Training/tree/main/orpo)]

## 🔄 即将到来
- Decoupled Clip and Dynamic sAmpling Policy Optimization (**DAPO**)
- 敬请期待

