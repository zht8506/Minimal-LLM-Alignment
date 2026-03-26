# SFT

This is a simplified implementation of SFT, supporting single-GPU and multi-GPU training. We use ``data/sft_en_demo.json`` as the demo sft dataset.

```
sft/
├── sft_dataset.py  # load SFT dataset
├── sft_train.py  # SFT training, single-GPU
├── sft_train_ngpu.py  # SFT training, multi-GPU
└── train.sh  # start SFT
```

Train LLM using SFT (including single-GPU and multi-GPU training versions):

```bash
bash train.sh
```
