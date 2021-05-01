import itertools
import numpy as np
from os.path import join, dirname
import pandas as pd
import joblib
import random
random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

from sklearn import svm
from sklearn.model_selection import KFold


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)

    def get_2(self, X, y):
        X_trans, y_trans = transform_pairwise(X, y)
        return super(RankSVM, self).predict(X_trans)

# Should be like that to handle model load/dump
if __name__ == "__main__":
    cv = KFold(n_splits=5)
    df = pd.read_csv(join(ROOT_DIR, f'baselines/data/rank.csv'), encoding= 'unicode_escape')
    # size = 10000
    # X = [
    #     df['title'][:size],
    #     df['vclaim'][:size],
    #     df['text'][:size]
    #     df['tdidf_text']
    # ]
    #
    # Y = [
    #     df['scores'][:size],
    #     df['iclaim_id'][:size]
    # ]

    X = [
        df['title'],
        df['vclaim'],
        df['text'],
        df['tfidf_title']
    ]

    Y = [
        df['scores'],
        df['iclaim_id']
    ]

    X = np.transpose(X)
    Y = np.transpose(Y)

    train, test = cv.split(X, Y).__next__()

    rank_svm = RankSVM(max_iter=3000).fit(X[train], Y[train])
    print('Performance of ranking ', rank_svm.score(X[test], Y[test]))

    joblib.dump(rank_svm, join(ROOT_DIR,"baselines/data/rank_svm-6.joblib"))