# Pretrained Models

Pretrained models are already released at [Baidu Netdisk](https://pan.baidu.com/s/1Z9aFGNq8q_tH0Zz_XpHHJg) (Extract Code: lsgp) and [Google Drive](https://drive.google.com/drive/folders/15uJQg9p74EJ8BlbBdpRVbYaE7vw2fITZ?usp=sharing). After your download the pretrained models, put it in the main folder, then execute the following commands to unzip it.

```bash
unzip stats.zip
```

The `stats` folder should have the following structure.

```
stats
├── bert
|   └── checkpoint.tar
├── bert_cased
|   └── checkpoint.tar
├── citation_bert
|   └── checkpoint.tar
├── citation_bert_cased
|   └── checkpoint.tar
└── vgae
    ├── checkpoint.tar
    ├── embedding.npy
    ├── specter_embedding.npy
    ├── train_pos_edge_list.csv
    ├── test_pos_edge_list.csv
    └── test_neg_edge_list.csv
```

where the `checkpoint.tar` is the pretrained models' checkpoints. The `specter_embedding.npy` is the paper embeddings extracted by the Specter model, and the `embedding.npy` is the embeddings extracted by our VGAE model. `train_pos_edge_list.csv`, `test_pos_edge_list.csv` and `test_neg_edge_list.csv` are the training and testing edges of the VGAE model.

**Note**. If you want to train your own model from begining, please keep `stats` folder clean, otherwise the training scripts will automatically load the checkpoints in the `stats` and continue training. If you want to fine-tune the model, you can put the checkpoints in the corresponding folder within the `stats` folder and then continue fine-tuning it.
