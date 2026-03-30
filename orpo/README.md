# ORPO

> [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

ORPO loss is the sum of standard SFT negative log-likelihood and a weighted odds ratio loss, which maximizes the odds ratio between chosen and rejected responses without a reference model.

This is a simplified implementation of ORPO, supporting single-GPU and multi-GPU training. We use ``data/dpo_en_demo.json`` as the demo dataset for ORPO.

```
orpo/
├── dpo_dataset.py  # load DPO dataset
├── orpo_train.py  # ORPO training, single-GPU
├── orpo_train_ngpu.py  # ORPO training, multi-GPU
└── train.sh  # start ORPO
```

Train LLM using ORPO (including single-GPU and multi-GPU training versions):

```bash
bash train.sh
```
