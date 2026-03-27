# GRPO

> [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

GRPO (Group Relative Policy Optimization) abandons the critic model in PPO, estimates the baseline from group rollout to calculate the advantage, adds KL divergence regularization directly to the loss, significantly reduces the memory and computational overhead of training.

This is a simplified implementation of GRPO. We use ``data/gsm8k_train_1of8.json`` and ``data/gsm8k_test_1of8.json``, which are both one-eighth samples of the GSM8K dataset, as the demo dataset for GRPO.

```
grpo/
├── gsm8k_dataset.py  # load gsm8k dataset
├── gsm8k_reward.py   # compute gsm8k reward
├── grpo_train.py  # GRPO training, single-GPU
└── train.sh  # start GRPO
```

Train LLM using GRPO (Single-GPU versions):

```bash
bash train.sh
```
