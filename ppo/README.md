# PPO

> [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

This is a simplified implementation of PPO, supporting single-GPU and multi-GPU training. We use ``data/gsm8k_train_1of8.json`` and ``data/gsm8k_test_1of8.json`` as the demo dataset for PPO.

```
ppo/
├── sft_dataset.py  # load SFT dataset
├── dft_train.py  # SFT training, single-GPU
├── dft_train_ngpu.py  # SFT training, multi-GPU
└── train.sh  # start DFT
```

Train LLM using DFT (including single-GPU and multi-GPU training versions):

```bash
bash train.sh
```
