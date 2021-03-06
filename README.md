![](assets/imgs/oh-my-papers.png)

# Oh-My-Papers: a Hybrid Context-aware Paper Recommendation System

[[Paper](assets/Oh-My-Papers%20a%20Hybrid%20Context-aware%20Paper%20Recommendation%20System.pdf)]

## Introduction

Current scholar search engine cannot recognize **"jargon"**, that is, specialized termilogy associated with a particular field or area of activity. For example, if you type "ResNet" in Google Scholar and other scholar search engine, you can not find the ResNet paper: "Deep Residual Learning for Image Recognition". In order to make the search more precisely, we build Oh-My-Papers, a hybrid context-aware citation recommendation system, as well as a scholar search engine.

We first point out that we can learn jargons from academic paper citation information. Since specialists usually use jargons such as "ResNet" in the academic paper writings, and the reference to the corresponding paper usually follows the jargon as illustrated below, citation information (especially citation context) of academic papers can help us to improve the searching results.

![](assets/imgs/citation.png)

In this work, we also create a large dataset recording citation information from papers of computer vision field in recent years. The dataset is ~10x larger than the biggest dataset from the previous works.

Besides paper recommendation, our search engine can also be regarded as a citation recommendation system. It can also perform auto-citation. We also provide related paper recommendation service based on our VGAE model.

We implement a simple front-end website, which has three functions: context searching, auto-citation and related paper recommendation. See the details from the demo repository [oh-my-papers-website](https://github.com/zhao-hr/oh-my-papers-website) for details.

## Requirements

Execute the following commands to install requirements.

```bash
pip install -r requirements.txt
```

You may need to manually install `pytorch-geometric` from its [official repo](https://github.com/rusty1s/pytorch_geometric) in order to install the correct version that is matched with your pytorch and cuda versions.

## Data Preparation

You can download full data at [Baidu Netdisk](https://pan.baidu.com/s/1N4zsWhieluiT3sC0FBASPw) (Extract Code: sc88) and [Google Drive](https://drive.google.com/drive/folders/1fxGBTk-2--HGsTNJXCnfRhQCuWlhTR8q?usp=sharing), and you need to put it into `data` folder. We also prepare a tiny dataset, which is a subset of the full dataset. If you want to prepare the data yourself, please see [docs/data_preparation.md](docs/data_preparation.md) for details.

## Pretrained Models

You can download full pretrained models at [Baidu Netdisk](https://pan.baidu.com/s/1Z9aFGNq8q_tH0Zz_XpHHJg) (Extract Code: lsgp) and [Google Drive](https://drive.google.com/drive/folders/1znpfVT0wi-uGjOczR9vT8GEkjY6LORHW?usp=sharing). After downloading the `stats.zip`, unzip it and put in under the main folder. Then, you can directly use it for inference and evaluation. Please see [docs/pretrained_models.md](docs/pretrained_models.md) for details.

## Models

Our repository includes three models:

- **Model 1**: VGAE model for related paper recommendation (ours).
- **Model 2**: Bert model for context-aware citation recommendation (baseline);
- **Model 3**: Citation-bert model for context-aware citation recommendation (ours).

Please see [docs/models.md](docs/models.md) for details.

## Configurations

Before training, evaluation and inference of the models, please set up your own configurations correctly. Please see [docs/configurations.md](docs/configurations.md) for details.

## Training (Optional)

If you have download our pretrained models, you can skip the following process. Before you start your own training from beginning, please keep the `stats` folder clean.

Execute the following commands to train model 1, model 2, model 3 respectively.

```bash
python train_vgae.py --cfg [Configuration Path]
```

```bash
python train_bert.py --cfg [Configuration Path]
```

```bash
python train_citation_bert.py --cfg [Configuration Path]
```

where `[Configuration Path]` is the path to your configuration file.

**Note**. If you want to train model 3, please train the model 1 first to generate the paper embeddings.

## Evaluation

If you want to evaluate the performance of our models, especially model 2 and model 3 (since model 1 almost has nothing to evaluate). For evaluation, please make sure that either you have downloaded pretrained models and put it in the correct place, or you have trained the models by yourselves.

Execute the following commands to evaluate model 2, model 3 respectively.

```bash
python eval_bert.py --cfg [Configuration Path]
```

```bash
python eval_citation_bert.py --cfg [Configuration Path]
```

where `[Configuration Path]` is the path to your configuration file.

Please see [docs/evaluation.md](docs/evaluation.md) for more details about the evaluation process.

## Inference

For inference, we create a class for the inference of each model. Please make sure that either you have downloaded pretrained models and put it in the correct place, or you have trained the models by yourselves.

Execute the following commands for inference of model 1, model 2 and model 3 respectively.

```bash
python inference_vgae.py --cfg [Configuration Path] --input [Input Path] --output [Output Path]
```

```bash
python inference_bert.py --cfg [Configuration Path] --input [Input Path] --output [Output Path]
```

```bash
python inferece_citation_bert.py --cfg [Configuration Path] --input [Input Path] --output [Output Path]
```

where `[Configuration Path]` is the path to your configuration file, and `[Input Path]` and `[Output Path]` is the path to the input file and the output file respectively.

**Note**. The input file and output file all has a json format.

Please see [docs/inference.md](docs/inference.md) for more details about the inference and the file format.

## Citations

```bibtex
@misc{fang2021ohmypapers,
  author =       {Hongjie Fang, Zhanda Zhu, Haoran Zhao},
  title =        {Oh-my-papers: a Hybrid Context-aware Citation Recommendation System},
  howpublished = {\url{https://github.com/Galaxies99/oh-my-papers}},
  year =         {2021}
}
```

## References

1. Science Parse: [official repo](https://github.com/allenai/science-parse/);
2. Transformers: [HuggingFace repo](https://github.com/huggingface/transformers/);
3. Specter: [official repo](https://github.com/allenai/specter);
4. Yang L, Zheng Y, Cai X, et al. A LSTM based model for personalized context-aware citation recommendation[J]. IEEE access, 2018, 6: 59618-59627.
5. Jeong C, Jang S, Park E, et al. A context-aware citation recommendation model with BERT and graph convolutional networks[J]. Scientometrics, 2020, 124(3): 1907-1922.
