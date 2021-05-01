# Nakov send
https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
https://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf

# Possible help:
https://www.quora.com/What-is-the-intuition-for-SVM-Rank-and-when-should-I-use-it
https://web.stanford.edu/class/cs276/handouts/lecture14-learning-ranking.pdf
https://www.researchgate.net/publication/229010211_SVM_Tutorial_Classification_Regression_and_Ranking

# SVMLight
https://stackoverflow.com/questions/25665569/load-svmlight-format-error
https://pypi.org/project/svmlight/

other possible interface:
http://tfinley.net/software/svmpython1/

# RankSVM source code
https://gist.github.com/agramfort/2071994

Based on LinearSVC of sklearn

Since Nakov paper uses rdf kernell, which is not
present in LinearSVC, we may try
to use:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

on top of: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# Running the project

## Create Dataset
python baselines/create-rank-svm-dataset.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/train.tsv --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b --lang=english

## Train the model
python baselines/ranksvm.py

## Execute Re-Ranking
python baselines/bm25-all-rank-svm.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/train.tsv --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b --lang=english