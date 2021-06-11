# Fact Checked Claim Detection

## About
This project solves the [CheckThat! 2021’s task](https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/tree/master/task2) for claim retrieval. 
The task is to rank a set of already verified claims by their relevance to a given input text, which contains a claim.

Our pipeline consists of the following steps:
1. Extract Twitter handles as names and split the hashtag.
2. Enrich the data by back translation, which provides our model with an augmented set of
examples.
3. Compute the S-BERT embeddings in order to assign scores by calculating the cosine
similarities between the input claims and the verified claim candidates.
4. Pass these scores as an input to our classifier.
5. Compute BM25 scores for the given input claim and the verified claims.
6. Pass the BM25 and S-BERT scores to RankSVM and obtain the final results.

The official results from our submission in the competition are MRR 0.795 for Task 2A and
0.336 for Task 2B, and include items 1–4 from the list above. After the official competition
deadline, we improved our results further to **0.909 for Task 2A** and to **0.559 for Task 2B** by
experimenting with steps 5 and 6.
   
## Requirements
Use the following command to install all necessary dependencies:
`pip install -r requirements.txt
`
## Installation
The project consists of multiple experiments that can be easily run through the CLI using the commands outlined below and the 
[pretrained models and embeddings](https://unisofiafaculty-my.sharepoint.com/:f:/g/personal/smartinovm_office365student_uni-sofia_bg/EjlhPsuoffxCuco5zS2kR8EBTmeNhASqae4x7PEAPsf_1g?e=DaejcQ)
that we have provided.

### S-BERT Experiment
#### Task 2A
`python baselines/sbert_2a.py --train-file-path=data/subtask-2a--english/qrels-train.tsv --dev-file-path=data/subtask-2a--english/qrels-dev.tsv --vclaims-dir-path=data/subtask-2a--english/vclaims --iclaims-file-path=data/subtask-2a--english/tweets-train-dev.tsv -ie embeddings/2a/iclaims_embeddings.npy -ve embeddings/2a/vclaims_embeddings.npy -te embeddings/2a/tclaims_embeddings.npy -de embeddings/2a/dclaims_embeddings.npy --lang=english --subtask=2a`

#### Task 2B
`python baselines/sbert_2b.py --train-file-path=data/subtask-2b--english/v1/train.tsv --dev-file-path=data/subtask-2b--english/v1/dev.tsv  --vclaims-dir-path=data/subtask-2b--english/politifact-vclaims --iclaims-file-path=data/subtask-2b--english/v1/iclaims.queries -ie embeddings/2b/iclaims_embeddings.npy -ve embeddings/2b/vclaims_embeddings.npy -te embeddings/2b/tclaims_embeddings.npy -de embeddings/2b/dclaims_embeddings.npy --lang=english --subtask=2b`

### BM-25 Experiment
#### Task 2A
`python baselines/bm25.py --train-file-path=data/subtask-2a--english/qrels-train.tsv --dev-file-path=data/subtask-2a--english/qrels-dev.tsv --vclaims-dir-path=data/subtask-2a--english/vclaims --iclaims-file-path=data/subtask-2a--english/tweets-train-dev.tsv --lang=english --subtask=2a`
#### Task 2B
`python baselines/bm25.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/train.tsv --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b --lang=english`

### RankSVM Experiment
Run all cells in the following Jupyter notebook `rankSVM_cornell/Generate_RankSVM_File.ipynb`
