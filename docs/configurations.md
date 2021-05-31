# Configurations

To create your own configuration files, please carefully read the details of this documents.

## Overview

The configuration files are all written in YAML format.

## Locations

All the default/provided configuration files are stored in the `configs` folder, in which

- `vgae.yaml` (default configuration) is the configuration file of model 1;
- `bert.yaml` (default configuration), `bert_cased.yaml` and `bert_uncased.yaml` are the configuration files of model 2, where `bert_cased.yaml` is the case-sensitive version of bert model, and both `bert.yaml` and `bert_uncased.yaml` is the case-insensitive version of bert model;
- `citation_bert.yaml` (default configuration), `citation_bert_cased.yaml` and `citation_bert_uncased.yaml` are the configuration files of model 3, where `citation_bert_cased.yaml` is the case-sensitive version of citation-bert model, and both `citation_bert.yaml` and `citation_bert_uncased.yaml` is the case-insensitive version of citation-bert model.
  
You can refer to the given configuration file to set up your own configurations. In the next sections we will introduce the possible items in the configuration of each model.

## VGAE (Model 1) Configurations

Here are the possible items in the configuration file of VGAE model (model 1).

- `max_epoch`: int, optional, default: 2000, the maximum of epoch num;
- `embedding_dim`: int, optional, default: 768, the embedding dimension of each node;
- `multigpu`: bool, optional, default: False, whether to multiple GPUs in training/inference;
- `adam_beta1`: float, optional, default: 0.9, the `beta1` parameter of Adam optimizer;
- `adam_beta2`: float, optional, default: 0.999, the `beta2` parameter of Adam optimizer;
- `learning_rate`: float, optional, default: 0.01, the initial learning rate of Adam optimizer;
- `specter_batch_size`: int, optional, default: 4, the batch size of the inference stage of specter model;
- `max_length`: int, optional, default: 512, the maximum length of context input to Bert model;
- `seq_len`: int, optional, default: 50, the maximum length of citation text;
- `end_year`: int, optional, default: 2020, the end year of the papers to train and evaluate;
- `frequency`: int, optional, default: 5, the minimum citations of a referenced papers to be counted;
- `stats_dir`: str, optional, default: 'stats/vgae', the directory of the statistics (including checkpoints and others);
- `data_path`: str, optional, default: 'data/citation.csv', the path to the data;
- `specter_embedding_filename`: str, optional, default: `specter_embeddings.npy`, the name of the file in which embeddings extracted by specter model is stored;
- `embedding_filename`: str, optional, default: `embedding.npy`, the name of the file in which embeddings extracted by our VGAE model is stored.

## Bert (Model 2) Configurations

Here are the possible items in the configuration file of Bert model (model 2).

- `max_epoch`: int, optional, default: 2000, the maximum of epoch num;
- `bert_cased`: bool, optional, default: False, whether the bert model is case-sensitive;
- `multigpu`: bool, optional, default: False, whether to multiple GPUs in training/inference;
- `adam_beta1`: float, optional, default: 0.9, the `beta1` parameter of AdamW optimizer;
- `adam_beta2`: float, optional, default: 0.999, the `beta2` parameter of AdamW optimizer;
- `adam_weight_decay`: float, optional, default: 0.01, the `weight_decay` parameter of AdamW optimizer;
- `adam_eps`: float, optional, default: 1e-6, the `eps` parameter of AdamW optimizer.
- `learning_rate`: float, optional, default: 0.01, the initial learning rate of Adam optimizer;
- `batch_size`: int, optional, default: 16, the batch size of the training/evaluation stage of the bert model;
- `max_length`: int, optional, default: 512, the maximum length of context input to Bert model;
- `seq_len`: int, optional, default: 50, the maximum length of citation text;
- `end_year`: int, optional, default: 2020, the end year of the papers to train and evaluate;
- `frequency`: int, optional, default: 5, the minimum citations of a referenced papers to be counted;
- `recall_K`, list of int, optional, default: [5, 10, 30, 50, 80], the Ks of the Recall@K metrics;
- `K`: int, optional, default: 10, the number of searching items in inference stage;
- `stats_dir`: str, optional, default: 'stats/vgae', the directory of the statistics (including checkpoints and others);
- `data_path`: str, optional, default: 'data/citation.csv', the path to the data.

## Citation-Bert (Model 3) Configurations

Here are the possible items in the configuration file of Citation-Bert model (model 3).

- `max_epoch`: int, optional, default: 2000, the maximum of epoch num;
- `multigpu`: bool, optional, default: False, whether to multiple GPUs in training/inference;
- `embedding_dim`: int, optional, default: 768, the embedding dimension of each node;
- `cosine_softmax_S`: float, optional, default: 1, the coefficient of the score combining process;
- `bert_cased`: bool, optional, default: False, whether the bert model is case-sensitive;
- `adam_beta1`: float, optional, default: 0.9, the `beta1` parameter of AdamW optimizer;
- `adam_beta2`: float, optional, default: 0.999, the `beta2` parameter of AdamW optimizer;
- `adam_weight_decay`: float, optional, default: 0.01, the `weight_decay` parameter of AdamW optimizer;
- `adam_eps`: float, optional, default: 1e-6, the `eps` parameter of AdamW optimizer.
- `learning_rate`: float, optional, default: 0.01, the initial learning rate of Adam optimizer;
- `batch_size`: int, optional, default: 16, the batch size of the training/evaluation stage of the bert model;
- `max_length`: int, optional, default: 512, the maximum length of context input to Bert model;
- `seq_len`: int, optional, default: 50, the maximum length of citation text;
- `end_year`: int, optional, default: 2020, the end year of the papers to train and evaluate;
- `frequency`: int, optional, default: 5, the minimum citations of a referenced papers to be counted;
- `recall_K`, list of int, optional, default: [5, 10, 30, 50, 80], the Ks of the Recall@K metrics;
- `K`: int, optional, default: 10, the number of searching items in inference stage;
- `stats_dir`: str, optional, default: 'stats/vgae', the directory of the statistics (including checkpoints and others);
- `data_path`: str, optional, default: 'data/citation.csv', the path to the data;
- `embedding_path`: str, optional, default: 'stats/vgae/specter_embedding.npy', the path to the embedding file of papers.
