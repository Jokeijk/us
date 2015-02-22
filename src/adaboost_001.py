import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter
from time import time
import sys
from sklearn.ensemble import AdaBoostClassifier


def output(Ytest,filename):
    out = pd.DataFrame({'Id':1+np.arange(Ytest.size),
        'Prediction':Ytest.astype(int)})
    out.to_csv(filename,index=False)
        
    
def get_data():
    train_file = '../data/kaggle_train_tf_idf.csv'
    test_file  = '../data/kaggle_test_tf_idf.csv'
    df_train = np.loadtxt(train_file,skiprows=1,delimiter=',')
    df_test  = np.loadtxt(test_file,skiprows=1,delimiter=',')
  
    X = df_train[:,1:-1]
    Y = df_train[:,-1]
    Xtest = df_test[:,1:]
    return X,Y,Xtest

def final_run(X,Y,Xtest):
    # 3 and 700 seems to be best
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
            algorithm="SAMME",
            n_estimators=700)
    clf.fit(X,Y)
    Ytest=clf.predict(Xtest)
    output(Ytest,'adaboost_001.csv')

X,Y,Xtest=get_data()
tic=time()
final_run(X,Y,Xtest)
toc=time()
print "time {} second ".format(toc-tic)

#for depth in [1,2]:
#    for n_est in range(100,1001,100):
#        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
#                algorithm="SAMME",
#                n_estimators=n_est)
#        score =  cross_validation.cross_val_score(clf,X,Y, cv=10, n_jobs=-1)
#        print "depth=", depth,"n_est=", n_est,"score=", score, " smean=", score.mean()

#for depth in range(3,7):
#    for n_est in range(100,1001,100):
#        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
#                algorithm="SAMME",
#                n_estimators=n_est)
#        score =  cross_validation.cross_val_score(clf,X,Y, cv=10, n_jobs=-1)
#        print "depth=", depth,"n_est=", n_est,"score=", score, " smean=", score.mean()
