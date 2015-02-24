# this file implements LASSO

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model, decomposition
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
# from dlinghu_SVM_001 import output, read_data, print_cv_scores
from dlinghu_functions import *


# tune parameters for LASSO
def lasso_tune_parameters(x_train, y_train):
    lasso = linear_model.Lasso()
    alpha_list = np.logspace(-8, 5, num=11)
    print "Entering GridSearchCV..."
    param_grid = dict(alpha=alpha_list)
    cv = StratifiedKFold(y=y_train, n_folds=3)
    grid = GridSearchCV(lasso, param_grid=param_grid, cv=cv)
    grid.fit(x_train, y_train)
    print("The best classifier is: ", grid.best_estimator_)
    clf = grid.best_estimator_
    return clf


def lasso_001():
    x_train, y_train, x_test = read_data()
    clf = lasso_tune_parameters(x_train, y_train)
    threshold, y_test_predict = find_reg_threshold(clf, x_train, y_train)
    # y_test_predict_raw = clf.predict(x_test)
    # y_test_predict = np.array(y_test_predict_raw)
    # y_test_predict[y_test_predict < threshold] = 0
    # y_test_predict[y_test_predict >= threshold] = 1
    # output(y_test_predict, 'LASSO_001.csv')  # alpha=1e-8, threshold=0.4


if __name__ == "__main__":
    lasso_001()