# DR.GRPO

> [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)

Dr. GRPO removes the length normalization and standard deviation in advantage estimation, improving token efficiency while preserving reasoning performance.

This is a simplified implementation of DR.GRPO. We use ``data/gsm8k_train_1of8.json`` and ``data/gsm8k_test_1of8.json``, which are both one-eighth samples of the GSM8K dataset, as the demo dataset for DR.GRPO.

```
dr.grpo/
├── gsm8k_dataset.py  # load gsm8k dataset
├── gsm8k_reward.py   # compute gsm8k reward
├── grpo_train.py  # DR.GRPO training, single-GPU
└── train.sh  # start DR.GRPO
```

Train LLM using DR.GRPO (Single-GPU versions):

```bash
bash train.sh
```
