# this file implements SVM with RBF kernel, uses grid search for the parameters
# C and Gamma

import numpy as np
import pandas as pd
import time
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


def output(y_test_predict, filename):
    out = pd.DataFrame({'Id': 1 + np.arange(y_test_predict.size),
                        'Prediction': y_test_predict})
    out.to_csv('../submit/%s' % filename, index=False)


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


# tune parameters for SVM
def svm_tune_parameter(x_train, y_train):
    # according to tests, the best C and Gamma should be in range 10-100
    c_range = 10.0 ** np.arange(-5, 4)
    gamma_range = 10.0 ** np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=c_range)
    cv = StratifiedKFold(y=y_train, n_folds=3)
    print "Entering GridSearchCV..."
    start_time = time.time()
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    print "Elapsed time = %s" % elapsed_time
    # print("The best classifier is: ", grid.best_estimator_)
    clf = grid.best_estimator_
    return clf


# print cross validation error
def print_cv_scores(clf, x_train, y_train, cv=5):
    scores = cross_validation.cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross validation errors: " % scores


def svm_001():
    x_train, y_train, x_test = read_data()

    clf = svm_tune_parameter(x_train, y_train)
    clf.fit(x_train, y_train)
    print "In-sample error = %s" % clf.score(x_train, y_train)

    print_cv_scores(clf, x_train, y_train, cv=5)

    y_test_predict = clf.predict(x_test)
    output(y_test_predict, 'SVM_001.csv')


if __name__ == "__main__":
    svm_001()