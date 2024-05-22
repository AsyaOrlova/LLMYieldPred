# Are Large Language Models the Best Estimators of Chemical Reaction Yields?

Chemical reaction yield, defined as the percentage of reactants turned into products, is the main criterion for selecting reaction conditions and evaluating success of a synthesis. Various machine learning (ML) models have been reported to predict reaction yields based on high-throughput experiment datasets. However, in the face of sparse and insufficient data typical for regular laboratory experiments, the performance and applicability of such models remain limited. More recently, the capabilities of large language models (LLMs) have been explored in predictive chemistry. Following up on this work, we investigate how LLMs perform in the generalized yield prediction task treated as a binary classification problem. In this regard, we engineer four different chemical reaction datasets to systematically evaluate performance of the top rated LLMs. We demonstrate that in the few-shot classification task LLMs outperform baseline approaches in F1-score up to 9\% and show competitive performance in terms of accuracy. Moreover, we observe superiority of ML models trained on LLM embeddings with the best average accuracy of 0.70 versus 0.67 achieved with current state-of-the-art approaches on the USPTO data. In this context, we discuss the potential of LLM embeddings to become the new state-of-the-art chemical reaction representations. Additionally, we share our empirical results on practical aspects of the few-shot LLM classifiers, such as the optimal size of the training set, and discuss peculiarities and prospects of the proposed methods.

![alt text](https://github.com/AsyaOrlova/llm_yield_pred/blob/main/assets/emb_gradient.jpg)

## :pushpin: Preparation of datasets

## :pushpin: Few-shot classification

Guidlines on performing few-shot classification experiments are provided in the `few_shot_classifier` folder.

## :pushpin: LLM embeddings

Guidlines on extracting reactions embeddings from text-embedding-3-large (OpenAI) and Mistral 7B (MistralAI) are provided in `openai_emb_api` and `mistralai_emb_api` folders.

## :pushpin: Training XGB on DRFPs and LLM embeddings

Scripts for XGB optimization are evaluation are provided in `xgb_drid_search` folder.

## :pushpin: Optimal training set investigation
