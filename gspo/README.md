# GSPO

> [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)

Unlike GRPO, which employs token-level importance ratios, GSPO defines its importance ratio based on sequence likelihood and conducts sequence-level clipping, rewarding, and optimization. Consequently, it exhibits superior training efficiency and performance relative to the GRPO algorithm, and notably stabilizes reinforcement learning training for Mixture-of-Experts (MoE) models.

This is a simplified implementation of GSPO. We use ``data/gsm8k_train_1of8.json`` and ``data/gsm8k_test_1of8.json``, which are both one-eighth samples of the GSM8K dataset, as the demo dataset for GSPO.

```
gspo/
├── gsm8k_dataset.py  # load gsm8k dataset
├── gsm8k_reward.py   # compute gsm8k reward
├── gspo_train.py  # GSPO training, single-GPU
└── train.sh  # start GSPO
```

Train LLM using GSPO (Single-GPU versions):

```bash
bash train.sh
```
