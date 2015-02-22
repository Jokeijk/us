# this file contains useful functions
import numpy as np
import pandas as pd
import time
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


def read_data():
    train_file = '../data/kaggle_train_tf_idf.csv'
    test_file = '../data/kaggle_test_tf_idf.csv'

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # for Y, it's important to reshape it since Y should have shape (4000, ), not (4000, 1)
    x_train = df_train.take(range(1, df_train.shape[1] - 1), axis=1).as_matrix()
    y_train = df_train.take([df_train.shape[1] - 1], axis=1).as_matrix().reshape(-1)
    x_test = df_test.take(range(1, df_train.shape[1] - 1), axis=1)
    return x_train, y_train, x_test


# print cross validation error
def print_cv_scores(clf, x_train, y_train, cv=5):
    scores = cross_validation.cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross validation errors: "
    print scores
    print "Average cross validation error: %s" % scores.mean()
    return scores.mean()


# print output to local file
def output(y_test_predict, filename):
    out = pd.DataFrame({'Id': 1 + np.arange(y_test_predict.size),
                        'Prediction': y_test_predict})
    out.to_csv('../submit/%s' % filename, index=False)


# tune parameters with GridSearchCV
# clf is some model of sklearn package
def tune_parameters(clf, x_train, y_train, param_grid):
    cv = StratifiedKFold(y=y_train, n_folds=3)
    print "Entering GridSearchCV..."
    start_time = time.time()
    grid = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid.fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    print "Elapsed time = %s" % elapsed_time
    print("The best classifier is: ", grid.best_estimator_)
    clf_best = grid.best_estimator_
    return clf_best
