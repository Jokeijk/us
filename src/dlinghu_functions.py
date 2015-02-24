# this file contains useful functions
import numpy as np
import pandas as pd
import time
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


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


# tune parameters for logistic regression
def lr_tune_parameter(x_train, y_train):
    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    pca.fit(x_train)
    cv = StratifiedKFold(y=y_train, n_folds=3)
    n_components = [20, 40, 80, 120, 160, 200]
    c_list = np.logspace(0, 8, num=9)
    print "Entering GridSearchCV..."

    grid = GridSearchCV(pipe,
                        dict(pca__n_components=n_components,
                             logistic__C=c_list),
                        cv=cv)
    grid.fit(x_train, y_train)
    print("The best classifier is: ", grid.best_estimator_)
    clf = grid.best_estimator_
    return clf


# apply AdaBoost on existing models
def ada_boost(clf, x_train, y_train, n_estimators, print_cv_error=0):
    print 'Entering AdaBoost algorithm...'
    start_time = time.time()
    clf_boost = AdaBoostClassifier(clf,
                                   algorithm="SAMME",
                                   n_estimators=n_estimators)
    clf_boost.fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    print "Elapsed time = %s" % elapsed_time
    print 'Done with AdaBoost!'
    if print_cv_error == 1:
        print 'Now calculating cv errors...'
        start_time = time.time()
        print 'For the original model:'
        print_cv_scores(clf, x_train, y_train)
        print 'For the boosted model:'
        print_cv_scores(clf_boost, x_train, y_train)
        elapsed_time = time.time() - start_time
        print "Elapsed time = %s" % elapsed_time
    else:
        print 'Now calculating in-sample error...'
        print "For the original model, in-sample score = %s" % clf.score(x_train, y_train)
        print "For the boosted model, in-sample score = %s" % clf_boost.score(x_train, y_train)
    return clf_boost


# find the best threshold
def find_reg_threshold(clf, x_train, y_train):
    clf.fit(x_train, y_train)
    threshold_list = np.arange(start=0.0, stop=1.0, step=0.05)
    threshold_best = 0
    accuracy_best = 0
    y_train_predict = clf.predict(x_train)
    for threshold in threshold_list:
        y_train_predict_binary = np.array(y_train_predict)
        y_train_predict_binary[y_train_predict_binary < threshold] = 0
        y_train_predict_binary[y_train_predict_binary >= threshold] = 1
        tmp_score = accuracy_score(y_train_predict_binary, y_train)
        # print '%s\t%s' % (threshold, tmp_score)
        print '{:>8} {:>8}'.format(*[threshold, tmp_score])
        if tmp_score > accuracy_best:
            accuracy_best = tmp_score
            threshold_best = threshold
            y_train_predict_binary_best = y_train_predict_binary
    print 'best threshold = %s' % threshold_best
    return threshold_best, y_train_predict_binary_best
