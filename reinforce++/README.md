# REINFORCE++

> [REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262v9)


This is a simplified implementation of REINFORCE++. We use ``data/gsm8k_train_1of8.json`` and ``data/gsm8k_test_1of8.json``, which are both one-eighth samples of the GSM8K dataset, as the demo dataset for REINFORCE++.

```
reinforce++/
├── gsm8k_dataset.py  # load gsm8k dataset
├── gsm8k_reward.py   # compute gsm8k reward
├── reinforce_plus_plus_train.py  # REINFORCE++ training, single-GPU
└── train.sh  # start REINFORCE++
```

Train LLM using REINFORCE++ (Single-GPU versions):

```bash
bash train.sh
```
