# DFT

This is a simplified implementation of [DFT](https://arxiv.org/abs/2508.05629), supporting single-GPU and multi-GPU training.

```
Dft/
├── sft_dataset.py  # load SFT dataset
├── dft_train.py  # SFT training, single-GPU
├── dft_train_ngpu.py  # SFT training, multi-GPU
└── train.sh  # start DFT
```

Train LLM with DFT:

```bash
bash train.sh
```
