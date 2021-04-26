# bm25.py

Contains experiment for a single value of vclaim.
title, vclaim, text

Run it with the following command from the main folder

python baselines/bm25.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/train.tsv --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b --lang=english


# bm25-all.py

Contains experiment for a combined values of vclaim

average value from title, vclaim, text 

Run it with the following command from the main folder

python baselines/bm25-all.py --train-file-path=baselines/v1/train.tsv --dev-file-path=baselines/v1/train.tsv --vclaims-dir-path=baselines/politifact-vclaims --iclaims-file-path=baselines/v1/iclaims.queries --subtask=2b --lang=english

# Experiments
https://docs.google.com/spreadsheets/d/1ddz8fIrt-MQdejRIMYEqdmJDCpmL1HrJwx28istm_Iw/edit?usp=sharing

# NOTES
Stemming, stopwords removal, and lowercasing were not applied and its a possible improvement

Make sure you use gensim 3.8.1, because bm25 its removed after this version