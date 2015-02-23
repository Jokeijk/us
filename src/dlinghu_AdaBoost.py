# apply AdaBoost on various models

import random
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from dlinghu_functions import *


x_train, y_train, x_test = read_data()
# apply AdaBoost on SVM (best SVM)
# c_range = 2.0 ** np.arange(6, 8)
# gamma_range = 2.0 ** np.arange(2, 5)
# param_grid = dict(gamma=gamma_range, C=c_range)
# svm = tune_parameters(SVC(), x_train, y_train, param_grid=param_grid)
# svm = SVC(C=128.0, gamma=8.0)
# svm = SVC(C=10000.0, gamma=8.0)
# svm_boost = ada_boost(svm, x_train, y_train, n_estimators=2, print_cv_error=1)
# y_test_predict = svm_boost.predict(x_test)
# output(y_test_predict, 'AdaBoost_SVM_RBF_001.csv')
# #############################################################
# test randomized SVM (weak learner)

# random_indices = random.sample(xrange(x_train.shape[0]), 200)
# x_train_random = x_train[random_indices, :]
# y_train_random = y_train[random_indices]
# svm = SVC(C=128.0, gamma=8.0)
# svm.fit(x_train_random, y_train_random)
# # print "In-sample score = %s" % svm.score(x_train, y_train)
# # print_cv_scores(svm, x_train, y_train)
# svm_boost = ada_boost(svm, x_train, y_train, n_estimators=10, print_cv_error=0)
# print "In-sample score = %s" % svm_boost.score(x_train, y_train)
# #############################################################
svm = SVC(C=128.0, gamma=8.0)
svm.fit(x_train, y_train)
print_cv_scores(svm, x_train, y_train)
svm_bagging = BaggingClassifier(svm)
svm_bagging.fit(x_train, y_train)
# test bagging
print "In-sample score = %s" % svm_bagging.score(x_train, y_train)
print_cv_scores(svm_bagging, x_train, y_train)

# lr_pca = lr_tune_parameter(x_train, y_train)
# lr_pca.fit(x_train, y_train)
# print_cv_scores(lr_pca, x_train, y_train)
# lr_pca_bagging = BaggingClassifier(lr_pca)
# lr_pca_bagging.fit(x_train, y_train)
# test bagging
# print "In-sample score = %s" % lr_pca_bagging.score(x_train, y_train)
# print_cv_scores(lr_pca_bagging, x_train, y_train)
lr = linear_model.LogisticRegression(C=10000.0)
lr.fit(x_train, y_train)
print_cv_scores(lr, x_train, y_train)
lr_bagging = BaggingClassifier(lr)
lr_bagging.fit(x_train, y_train)
print_cv_scores(lr_bagging, x_train, y_train)
y_test_predict_lr_bagging = lr_bagging.predict(x_test)

# from bagging to boosting
print 'From bagging to boosting...'
start_time = time.time()
estimator_list = [svm_bagging, lr_bagging, svm, lr]
# meta_boost = AdaBoostClassifier(algorithm="SAMME")
# meta_boost.estimators_ = estimator_list
meta_boost = AdaBoostClassifier(svm_bagging, algorithm='SAMME')
meta_boost.fit(x_train, y_train)
print "In-sample score = %s" % meta_boost.score(x_train, y_train)
print_cv_scores(meta_boost, x_train, y_train)
elapsed_time = time.time() - start_time
print "Elapsed time = %s" % elapsed_time