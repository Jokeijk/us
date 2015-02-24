# this file gives predictions on the training data

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
from sklearn.ensemble import GradientBoostingClassifier
from dlinghu_functions import *
from dlinghu_Ensemble import list_to_array, manual_bagging


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
if __name__ == "__main__":
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
    y_test_predict_list = manual_bagging(estimators, x_train, y_train, x_test, bagging=False)
    y_test_predict_array = list_to_array(y_test_predict_list)
    np.savetxt("y_test_predict_array.csv", y_test_predict_array)
    print 'File saved!'