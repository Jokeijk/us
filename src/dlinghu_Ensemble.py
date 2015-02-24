# ensemble learning

import random
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from dlinghu_functions import *
from sklearn.ensemble import GradientBoostingClassifier


# y_predict_collection = np.array(x_test.shape[0], len(estimators) * n_bootstrap)
# y_test_predict_list = []
# y_train


# if bagging=False, then this function returns list of predictions for all estimator
# in given estimators
def manual_bagging(estimators, x_train, y_train, x_test, bagging=True, n_bootstrap=10):
    y_predict_list = []
    n_train = x_train.shape[0]
    if bagging:
        for n in xrange(n_bootstrap):
            random_indices = np.random.choice(np.arange(n_train), size=n_train)
            x_train_sample = x_train[random_indices, :]
            y_train_sample = y_train[random_indices]
            for (name, estimator) in estimators:
                print '%s-th bootstrapping for %s...' % (n, name)
                estimator.fit(x_train_sample, y_train_sample)
                y_predict_list.append(estimator.predict(x_test))
    else:
        for (name, estimator) in estimators:
            print 'predicting via %s...' % name
            estimator.fit(x_train, y_train)
            y_predict_list.append(estimator.predict(x_test))
    return y_predict_list


# convert list of arrays to a 2d array
def list_to_array(list):
    array_2d = np.zeros((list[0].shape[0], len(list)))
    for i in xrange(len(list)):
        array_2d[:, i] = list[i]
    return array_2d


# this function sets a validation dataset to run regression on the predictions to get
# the best linear coefficients to combine models, then combine their predictions on
# the test dataset.
def reg_ensemble(estimators, valid_ratio, x_train, y_train, x_test, n_bootstrap=10):
    x_train_sub, x_valid, y_train_sub, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio)
    y_valid_predict_list = manual_bagging(estimators, x_train_sub, y_train_sub, x_valid, bagging=False)
    y_valid_predict_array = list_to_array(y_valid_predict_list)
    clf = LinearRegression()
    threshold, y_valid_predict_ensemble = find_reg_threshold(clf, y_valid_predict_array, y_valid)
    print "Ensembled score = %s" % accuracy_score(y_valid_predict_ensemble, y_valid)
    print "for comparison, here are single predictor performance:"
    for i in xrange(y_valid_predict_array.shape[1]):
        print "score for %s-th predictor = %s" \
              % (i, accuracy_score(y_valid_predict_array[:, i], y_valid))
    # now predict test data
    y_test_predict_list = manual_bagging(estimators, x_train, y_train, x_test, bagging=False)
    y_test_predict_array = list_to_array(y_test_predict_list)
    clf.fit(y_valid_predict_array, y_valid)
    y_test_predict_raw = clf.predict(y_test_predict_array)  # get regression result
    y_test_predict = np.zeros((x_test.shape[0], 1))
    y_test_predict[y_test_predict_raw >= threshold] = 1  # use threshold to get binary prediction
    return y_test_predict


def ensemble(clf, estimators, valid_ratio, x_train, y_train, x_test):
    x_train_sub, x_valid, y_train_sub, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio)
    y_valid_predict_list = manual_bagging(estimators, x_train_sub, y_train_sub, x_valid, bagging=False)
    y_valid_predict_array = list_to_array(y_valid_predict_list)
    threshold, y_valid_predict_ensemble = find_reg_threshold(clf, y_valid_predict_array, y_valid)
    print "Ensembled score = %s" % accuracy_score(y_valid_predict_ensemble, y_valid)
    print "for comparison, here are single predictor performance:"
    for i in xrange(y_valid_predict_array.shape[1]):
        print "score for %s-th predictor = %s" \
              % (i, accuracy_score(y_valid_predict_array[:, i], y_valid))
    # now predict test data
    y_test_predict_list = manual_bagging(estimators, x_train, y_train, x_test, bagging=False)
    y_test_predict_array = list_to_array(y_test_predict_list)
    clf.fit(y_valid_predict_array, y_valid)
    y_test_predict = clf.predict(y_test_predict_array)  # get ensemble result
    return y_test_predict


# def majority_vote(y_predict_list):
# y_predict_cumulative = np.zeros((y_predict_list[0].shape[0], 1))
# y_predict = np.zeros((y_predict_list[0].shape[0], 1))
#     for y_predict_sample in y_predict_list:
#         y_predict_cumulative += y_predict_sample.reshape(-1, 1)
#     y_predict[y_predict_cumulative > (len(y_predict_list) / 2)] = 1
#     return y_predict

x_train, y_train, x_test = read_data()
svm = SVC(C=128.0, gamma=8.0)
lr = linear_model.LogisticRegression(C=10000.0)
knn = KNeighborsClassifier(n_neighbors=20)
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, n_estimators=300,
                            max_features=300, random_state=0)
dt_boosted = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, max_features=100),
                                algorithm="SAMME",
                                n_estimators=700)




# ###################################################################
# 0.9430 adaboost_005.py

clf1 = AdaBoostClassifier(
    DecisionTreeClassifier(min_samples_split=500),
    algorithm="SAMME", n_estimators=2000)

# 0.9365 gradient_boost_newnew.py

clf2 = GradientBoostingClassifier(
    n_estimators=2000,
    max_depth=3,
    max_features=None,
    learning_rate=0.08,
    random_state=0)

# 0.93750, adaboost_003.py
clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, max_features=100),
                          algorithm="SAMME",
                          n_estimators=700)

# 0.9395, adaboost_001.py
clf4 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                          algorithm="SAMME",
                          n_estimators=700)
###################################################################
# estimators = [('SVM', svm), ('Logistic', lr), ('kNN-20', knn),
# ('DT-boosted', dt_boosted)]
# estimators = [('SVM', svm), ('Logistic', lr), ('kNN-20', knn), ('RF', rf)]
estimators = [('SVM', svm), ('Logistic', lr), ('kNN-20', knn), ('RF', rf),
              ('clf1', clf1), ('clf2', clf2), ('clf3', clf3), ('clf4', clf4)]

n_bootstrap = 10
valid_ratio = 0.15  # use 15% training data as validation data
# n_train = x_train.shape[0]
# n_test = x_test.shape[0]
# y_test_predict = reg_ensemble(estimators, valid_ratio, x_train, y_train, x_test)
y_test_predict = ensemble(linear_model.LogisticRegression(),
                          estimators, valid_ratio, x_train, y_train, x_test)
output(y_test_predict.reshape(-1), 'dlinghu_Ensemble_002.csv')
