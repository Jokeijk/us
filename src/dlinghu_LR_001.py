# this file implements logistic regression

from dlinghu_functions import *


def lr_001():
    x_train, y_train, x_test = read_data()
    clf = lr_tune_parameter(x_train, y_train)
    print_cv_scores(clf, x_train, y_train)
    y_test_predict = clf.predict(x_test)
    # output(y_test_predict, 'LR_001.csv')# n_components=200, C=10000


if __name__ == "__main__":
    lr_001()