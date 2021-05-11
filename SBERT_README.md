### External files
- [Embeddings](https://unisofiafaculty-my.sharepoint.com/:f:/g/personal/smartinovm_office365student_uni-sofia_bg/EjlhPsuoffxCuco5zS2kR8EBTmeNhASqae4x7PEAPsf_1g?e=DaejcQ)

### Run
#### Task 2A
`python baselines/sbert_2a.py --train-file-path=data/subtask-2a--english/qrels-train.tsv --dev-file-path=data/subtask-2a--english/qrels-dev.tsv --vclaims-dir-path=data/subtask-2a--english/vclaims --iclaims-file-path=data/subtask-2a--english/tweets-train-dev.tsv -ie embeddings/2a/iclaims_embeddings.npy -ve embeddings/2a/vclaims_embeddings.npy -te embeddings/2a/tclaims_embeddings.npy -de embeddings/2a/dclaims_embeddings.npy --lang=english --subtask=2a`

#### Task 2B
`python baselines/classifier.py --train-file-path=data/subtask-2b--english/v1/train.tsv --dev-file-path=data/subtask-2b--english/v1/dev.tsv  --vclaims-dir-path=data/subtask-2b--english/politifact-vclaims --iclaims-file-path=data/subtask-2b--english/v1/iclaims.queries -ie embeddings/2b/iclaims_embeddings.npy -ve embeddings/2b/vclaims_embeddings.npy -te embeddings/2b/tclaims_embeddings.npy -de embeddings/2b/dclaims_embeddings.npy --lang=english --subtask=2b`
