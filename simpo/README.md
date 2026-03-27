# SIMPO

> [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

SimPO takes the average log probability of a sequence as an implicit reward to dispense with a reference model and adds a target reward margin to the Bradley-Terry objective to boost performance.

This is a simplified implementation of SimPO, supporting single-GPU and multi-GPU training. We use ``data/dpo_en_demo.json`` as the demo dataset for DFT.

```
simpo/
├── dpo_dataset.py  # load SimPO dataset
├── simpo_train.py  # SimPO training, single-GPU
├── simpo_train_ngpu.py  # SimPO training, multi-GPU
└── train.sh  # start SimPO
```

Train LLM using SimPO (including single-GPU and multi-GPU training versions):

```bash
bash train.sh
```
