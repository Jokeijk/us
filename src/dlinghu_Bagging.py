# this file tests bagging on various algorithms

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from dlinghu_functions import *


x_train, y_train, x_test = read_data()
svm = SVC(C=128.0, gamma=8.0)
svm.fit(x_train, y_train)
print_cv_scores(svm, x_train, y_train)
#########################################################
# test bagging sample ratio, without replacement
for max_sample in np.arange(0.1, 1.0, 0.1):
    print 'max_sample ratio = %s' % max_sample
    svm_bagging = BaggingClassifier(svm, bootstrap=False, max_samples=max_sample, n_estimators=50)
    svm_bagging.fit(x_train, y_train)
    # test bagging
    print "In-sample score = %s" % svm_bagging.score(x_train, y_train)
    print_cv_scores(svm_bagging, x_train, y_train)
#########################################################
svm_bagging = BaggingClassifier(svm, bootstrap=True, n_estimators=50)
svm_bagging.fit(x_train, y_train)
print_cv_scores(svm_bagging, x_train, y_train)