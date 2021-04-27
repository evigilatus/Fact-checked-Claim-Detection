import itertools
import numpy as np
from os.path import join, dirname, basename, exists
import pandas as pd
import joblib
import random
random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

from sklearn import svm, linear_model
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
        return super(RankSVM, self).predict(X_trans) == y_trans

    def get_2(self, X, y):
        X_trans, y_trans = transform_pairwise(X, y)
        return super(RankSVM, self).predict(X_trans)

def run_test():
    # as showcase, we will create some non-linear data
    # and print the performance of ranking vs linear regression

    # np.random.seed(1)
    # n_samples, n_features = 300, 5
    # true_coef = np.random.randn(n_features)
    # # print(true_coef)
    # X = np.random.randn(n_samples, n_features)
    # # print(X)
    # noise = np.random.randn(n_samples) / np.linalg.norm(true_coef)
    # y = np.dot(X, true_coef)
    # # print(y)
    # y = np.arctan(y)  # add non-linearities
    # y += .1 * noise  # add noise
    # Y = np.c_[y, np.mod(np.arange(n_samples), 5)]  # add query fake id
    # # print(Y)
    cv = KFold(n_splits=5)
    # columns = ['iclaim_id', 'vclaim_id', 'title', 'vclaim', 'text', 'scores'])
    df = pd.read_csv(join(ROOT_DIR, f'baselines/data/rank-encoded.csv'))
    # df['iclaim_encoded'] = df['iclaim_id'].apply(lambda x: int("".join([str(ord(item)) for item in x])))
    # df.to_csv(join(ROOT_DIR, f'baselines/data/rank-encoded.csv'))
    size = 100000
    X = [
        df['title'][:size],
        df['vclaim'][:size],
        df['text'][:size]
    ]

    Y = [
        df['scores'][:size],
        df['iclaim_encoded'][:size]
    ]

    X = np.transpose(X)
    Y = np.transpose(Y)
    # print(np.shape(X))
    # print(np.shape(Y))
    # print(X[:10])
    # print(Y[:10])
    # print(df.sample(100))
    # print(df.describe())

    train, test = cv.split(X, Y).__next__()

    # make a simple plot out of it
    import pylab as pl

    # pl.scatter(np.dot(X, true_coef), y)
    # pl.title('Data to be learned')
    # pl.xlabel('<X, coef>')
    # pl.ylabel('y')
    # pl.show()

    # print the performance of ranking
    rank_svm = RankSVM().fit(X[train], Y[train])

    joblib.dump(rank_svm, join(ROOT_DIR,"baselines/data/rank_svm.joblib"))
    print("scores")
    loaded_rank_svm = joblib.load(join(ROOT_DIR, "baselines/data/rank_svm.joblib"))
    print(loaded_rank_svm.decision_function(X[test]))

    # print('Performance of ranking ', rank_svm.get_2(X[test], Y[test]))

    # and that of linear regression
    # ridge = linear_model.RidgeCV(fit_intercept=True)
    # ridge.fit(X[train], y[train])
    # X_test_trans, y_test_trans = transform_pairwise(X[test], y[test])
    # score = np.mean(np.sign(np.dot(X_test_trans, ridge.coef_)) == y_test_trans)
    # print('Performance of linear regression ', score)


run_test()