### External files
- [Embeddings](https://unisofiafaculty-my.sharepoint.com/:f:/g/personal/smartinovm_office365student_uni-sofia_bg/EjlhPsuoffxCuco5zS2kR8EBTmeNhASqae4x7PEAPsf_1g?e=DaejcQ)

### Run
#### Task 2A
`python baselines/sbert_2a.py --train-file-path=data/subtask-2a--english/qrels-train.tsv --dev-file-path=data/subtask-2a--english/qrels-dev.tsv  --vclaims-dir-path=data/subtask-2a--english/vclaims --iclaims-file-path=data/subtask-2a--english/tweets-train-dev.tsv -ie embeddings/iclaims_embeddings.npy -ve embeddings/vclaims_embeddings.npy -te embeddings/tclaims_embeddings.npy -de embeddings/dclaims_embeddings.npy --lang=english --subtask=2a`
#### Task 2B
`python baselines/classifier.py --train-file-path=data/subtask-2b--english/v1/train.tsv --dev-file-path=data/subtask-2b--english/v1/dev.tsv  --vclaims-dir-path=data/subtask-2b--english/politifact-vclaims --iclaims-file-path=data/subtask-2b--english/v1/iclaims.queries -ie embeddings/iclaims_embeddings.npy -ve embeddings/vclaims_embeddings.npy -te embeddings/tclaims_embeddings.npy -de embeddings/dclaims_embeddings.npy --lang=english --subtask=2b`
