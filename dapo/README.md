# DAPO

> [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)

DAPO introduces Clip-Higher, Token-Level Policy Gradient Loss, Overlong Reward Shaping and Dynamic Sampling to improve GRPO. This project implements the first three functions.

This is a simplified implementation of DAPO. We use ``data/gsm8k_train_1of8.json`` and ``data/gsm8k_test_1of8.json``, which are both one-eighth samples of the GSM8K dataset, as the demo dataset for DAPO.

```
dapo/
├── gsm8k_dataset.py  # load gsm8k dataset
├── gsm8k_reward.py   # compute gsm8k reward
├── dapo_train.py  # DAPO training, single-GPU
└── train.sh  # start DAPO
```

Train LLM using DAPO (Single-GPU versions):

```bash
bash train.sh
```
