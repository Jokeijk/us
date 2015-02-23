# this file implements SVM with RBF kernel, uses grid search for the parameters
# C and Gamma

from sklearn.svm import SVC
from dlinghu_functions import *


# tune parameters for SVM
def svm_tune_parameter(x_train, y_train):
    # according to tests, the best (C, Gamma) should be (100, 10)
    c_range = 2.0 ** np.arange(6, 8)
    gamma_range = 2.0 ** np.arange(3, 5)
    param_grid = dict(gamma=gamma_range, C=c_range)
    cv = StratifiedKFold(y=y_train, n_folds=3)
    print "Entering GridSearchCV..."
    start_time = time.time()
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    print "Elapsed time = %s" % elapsed_time
    print("The best classifier is: ", grid.best_estimator_)
    clf = grid.best_estimator_
    return clf


def svm_001():
    x_train, y_train, x_test = read_data()

    clf = svm_tune_parameter(x_train, y_train)
    clf.fit(x_train, y_train)
    print "In-sample score = %s" % clf.score(x_train, y_train)

    print_cv_scores(clf, x_train, y_train, cv=5)

    y_test_predict = clf.predict(x_test)
    # output(y_test_predict, 'SVM_001.csv') # C=100, Gamma=10
    # output(y_test_predict, 'SVM_002.csv') # C=128, Gamma=8


if __name__ == "__main__":
    svm_001()