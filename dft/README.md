# DFT

> [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](https://arxiv.org/abs/2508.05629)

The paper propose the **Dynamic Fine-Tuning (DFT)**, which stabilizes gradient updates for each token by dynamically rescaling the objective function using the token’s probability.

This is a simplified implementation of DFT, supporting single-GPU and multi-GPU training. We use ``data/sft_en_demo.json`` as the demo dataset for DFT.

```
dft/
├── sft_dataset.py  # load SFT dataset
├── dft_train.py  # SFT training, single-GPU
├── dft_train_ngpu.py  # SFT training, multi-GPU
└── train.sh  # start DFT
```

Train LLM using DFT (including single-GPU and multi-GPU training versions):

```bash
bash train.sh
```
