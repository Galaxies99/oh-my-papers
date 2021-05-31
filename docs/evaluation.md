# Evaluation

We support the evaluation of bert model (model 2) and citation-bert model (model 3). Currently we calculate the following metrics:

1. mAP: calculating mAP is relatively slow, if you want to speed up the evaluation, just ignoring mAP by changing
   
   ```python
   recorder = ResultRecorder(paper_num, include_mAP = True, recall_K = RECALL_K)
   ```

   to

   ```python
   recorder = ResultRecorder(paper_num, include_mAP = False, recall_K = RECALL_K)
   ```

   in the evaluation program (`eval_bert.py` and `eval_citation_bert.py`).
2. mRR: the mean reciprocal rank;
3. Recall@K: the hit rate of the top-K recommendations. Here the values of `K` is specified in `recall_K` item in the configuration file.
