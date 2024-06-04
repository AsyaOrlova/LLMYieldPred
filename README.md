# Are Large Language Models the Best Estimators of Chemical Reaction Yields?

Chemical reaction yield, defined as the percentage of reactants turned into products, is the main criterion for selecting reaction conditions and evaluating success of a synthesis. Various machine learning (ML) models have been reported to predict reaction yields based on high-throughput experiment datasets. However, in the face of sparse and insufficient data typical for regular laboratory experiments, the performance and applicability of such models remain limited. More recently, the capabilities of large language models (LLMs) have been explored in predictive chemistry. Following up on this work, we investigate how LLMs perform in the generalized yield prediction task treated as a binary classification problem. In this regard, we engineer four different chemical reaction datasets to systematically evaluate performance of the top rated LLMs. We demonstrate that in the few-shot classification task LLMs outperform baseline approaches in F1-score up to 9\% and show competitive performance in terms of accuracy. Moreover, we observe superiority of ML models trained on LLM embeddings with the best average accuracy of 0.70 versus 0.67 achieved with current state-of-the-art approaches on the USPTO data. In this context, we discuss the potential of LLM embeddings to become the new state-of-the-art chemical reaction representations. Additionally, we share our empirical results on practical aspects of the few-shot LLM classifiers, such as the optimal size of the training set, and discuss peculiarities and prospects of the proposed methods.

![alt text](https://github.com/AsyaOrlova/llm_yield_pred/blob/main/assets/emb_gradient.jpg)

## :pushpin: Preparation of datasets
All notebooks regarding the datasets preparation process can be found in the ```data``` folder.
### USPTO
We used the Schwaller's version of the USPTO dataset, which was originally suggested in [this](https://pubs.rsc.org/en/content/articlelanding/2018/SC/C8SC02339E) paper. The dataset itself can be found [here](https://ibm.ent.box.com/v/ReactionSeq2SeqDataset). We additionally preprocessed the dataset as suggested in ```1_uspto_processing.ipynb```. Then, we created two smaller datasets: USPTO-R and USPTO-C (see ```3_uspto_datasets_preparation.ipynb```).
### ORD
We parsed the ORD dataset using ordschema API. The dataset was preprocessed as suggested in ```2_ord_processing.ipynb```. The preparation of ORD-R and ORD-C datasets is described in ```4_ord_datasets_preparation.ipynb```.

Reactions from each dataset were converted into text descriptions as suggested in ```5_smiles_to_iupac.ipynb```.
The datasets for the experiments on the optimal training subset size were obtained as suggested in ```6_train_size_datasets_preparation.ipynb```.

## :pushpin: Few-shot classification

Guidlines on performing few-shot classification experiments are provided in the `few_shot_classifier` folder.

## :pushpin: LLM embeddings

Guidlines on extracting reactions embeddings from text-embedding-3-large (OpenAI) and Mistral 7B (MistralAI) are provided in `openai_emb_api` and `mistralai_emb_api` folders.

## :pushpin: Training XGB on DRFPs and LLM embeddings

Scripts for XGB optimization are evaluation are provided in `xgb_drid_search` folder.

## :pushpin: Optimal training set investigation
