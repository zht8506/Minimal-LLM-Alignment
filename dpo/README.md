# DPO

This is a simplified implementation of DPO, supporting single-GPU and multi-GPU training. We use ``data/dpo_en_demo.json`` as the demo dataset for DPO.

```
dpo/
├── dpo_dataset.py  # load DPO dataset
├── dpo_train.py  # DPO training, single-GPU
├── dpo_train_ngpu.py  # DPO training, multi-GPU
└── train.sh  # start DPO
```

Train LLM using DPO (including single-GPU and multi-GPU training versions):

```bash
bash train.sh
```
