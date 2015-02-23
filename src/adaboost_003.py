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
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,max_features=100),
            algorithm="SAMME",
            n_estimators=700)
    clf.fit(X,Y)
    Ytest=clf.predict(Xtest)
    output(Ytest,'adaboost_003.csv')

def report(grid_scores, n_top=5):
    params = None
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Parameters with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
              score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

        if params == None:
            params = score.parameters

    return params

def dogridsearch(X,Y,param_space,clf,cv):
    grid_search = GridSearchCV(clf,param_space,verbose=10l,cv=cv,n_jobs=-1)
    start = time()
    grid_search.fit(X,Y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))
    best = report(grid_search.grid_scores_)


def backcheck_case(X,Y):
    depth=3
    n_est=700
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
        algorithm="SAMME",
        n_estimators=n_est,random_state=0)
    score =  cross_validation.cross_val_score(clf,X,Y, cv=5, n_jobs=-1)
    print "depth=", depth,"n_est=", n_est,"score=", score, " smean=", score.mean()

def cross_val(X,Y):
    clf = AdaBoostClassifier(DecisionTreeClassifier())
    param_grid={
            "n_estimators":range(500,1201,100),
            "base_estimator__max_depth":[1,2,3,4],
            "base_estimator__max_features":['auto',100,300]
            } # 700 3 300
    param_grid={
            "n_estimators":[700,750],
            "base_estimator__max_depth":[3],
            "base_estimator__max_features":range(200,331,10)
            }
    param_grid={
            "n_estimators":range(700,790,3),
            "base_estimator__max_depth":[3],
            "base_estimator__max_features":[300]
            }
    param_grid={
            "n_estimators":[721],
            "base_estimator__max_depth":[3],
            "base_estimator__max_features":['auto',300]
            }
    param_grid={
            "n_estimators":[700],
            "base_estimator__max_depth":[3],
            "algorithm":["SAMME"],
            "base_estimator__max_features":['auto',100,300]
            }
    dogridsearch(X,Y,param_grid,clf,cv=10)

def main():
    X,Y,Xtest=get_data()
    tic=time()
    #backcheck_case(X,Y)
    #cross_val(X,Y)
    final_run(X,Y,Xtest)
    toc=time()
    print "time {} second ".format(toc-tic)

if __name__ == '__main__':
    main()
