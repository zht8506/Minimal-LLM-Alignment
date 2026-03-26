# PPO

> [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

This is a simplified implementation of PPO. We use ``data/gsm8k_train_1of8.json`` and ``data/gsm8k_test_1of8.json``, which are both one-eighth samples of the GSM8K dataset, as the demo dataset for PPO.

```
ppo/
├── gsm8k_dataset.py  # load gsm8k dataset
├── gsm8k_reward.py   # compute gsm8k reward
├── ppo_train.py  # PPO training, single-GPU
└── train.sh  # start PPO
```

Train LLM using PPO (Single-GPU versions):

```bash
bash train.sh
```
