# RLOO

> [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)

RLOO (REINFORCE Leave-One-Out) is an improved policy gradient RL method that uses the average reward of other samples in a batch as baseline, achieving Monte Carlo policy updates with reduced variance.

This is a simplified implementation of RLOO. We use ``data/gsm8k_train_1of8.json`` and ``data/gsm8k_test_1of8.json``, which are both one-eighth samples of the GSM8K dataset, as the demo dataset for RLOO.

```
grpo/
├── gsm8k_dataset.py  # load gsm8k dataset
├── gsm8k_reward.py   # compute gsm8k reward
├── rloo_train.py  # RLOO training, single-GPU
└── train.sh  # start RLOO
```

Train LLM using RLOO (Single-GPU versions):

```bash
bash train.sh
```
