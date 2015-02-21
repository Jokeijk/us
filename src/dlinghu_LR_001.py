# this file implements logistic regression

import numpy as np
import pandas as pd
import time
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from dlinghu_SVM_001 import output, read_data, print_cv_scores


def lr_tune_parameter(x_train, y_train):
    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    pca.fit(x_train)
    n_components = [20, 40, 80, 120, 160, 200]
    Cs = np.logspace(0, 8, num=9)
    print "Entering GridSearchCV..."
    grid = GridSearchCV(pipe,
                        dict(pca__n_components=n_components,
                             logistic__C=Cs))
    grid.fit(x_train, y_train)
    print("The best classifier is: ", grid.best_estimator_)
    clf = grid.best_estimator_
    return clf


def lr_001():
    x_train, y_train, x_test = read_data()
    clf = lr_tune_parameter(x_train, y_train)
    print_cv_scores(clf, x_train, y_train)
    y_test_predict = clf.predict(x_test)
    # output(y_test_predict, 'LR_001.csv')


if __name__ == "__main__":
    lr_001()