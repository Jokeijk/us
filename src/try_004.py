import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter
from time import time


def output(Ytest,filename):
    out = pd.DataFrame({'Id':1+np.arange(Ytest.size),'Prediction':Ytest})
    out.to_csv(filename,index=False)
        
    
def get_data():
    train_file = '../data/kaggle_train_tf_idf.csv'
    test_file  = '../data/kaggle_test_tf_idf.csv'
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)  
  
    X = df_train.values[:,1:-1]
    Y = df_train.values[:,-1]
    Xtest = df_test.values[:,1:]
    return X,Y,Xtest

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

def dogridsearch(X,Y,param_space):
    clf = RandomForestClassifier(oob_score=True,n_jobs=-1) 
    grid_search = GridSearchCV(clf,param_space,verbose=10l)
    start = time()
    grid_search.fit(X,Y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))      
    best = report(grid_search.grid_scores_)

def onescore(X,Y):
    clf = RandomForestClassifier(oob_score=True,n_jobs=-1, n_estimators=1000, max_features=300)
    clf.fit(X,Y)
    print "oob_score = ",  clf.oob_score_
    print clf.get_params()


X,Y,Xtest=get_data()
grid_test0 = { "n_estimators"      : [1000],
               "max_features"      : [300,500] }
dogridsearch(X,Y,grid_test0)

#tic = time()
#onescore(X,Y)
#toc = time()
#print "time= ",toc-tic

#grid_test0 = { "n_estimators"      : [400],
#               "criterion"         : ["gini"],
#               "max_features"      : [300],
#               "max_depth"         : [100,200],
#               "min_samples_split" : [2],
#               "min_samples_leaf"  : [1]}
#               
#grid_test1 = { "n_estimators"      : [100,200],
#               "criterion"         : ["gini"],
#               "max_features"      : [100,200,300],
#               "max_depth"         : [10,20],
#               "min_samples_split" : [2,5],
#               "min_samples_leaf"  : [1,5]}
#               
#grid_test2 = { "n_estimators"      : np.rint(np.linspace(100, 401, 100)).astype(int),
#                 "criterion"         : ["gini", "entropy"],
#                 "max_features"      : np.rint(np.linspace(100, 501, 100)).astype(int),
#                 "min_samples_split" : np.rint(np.linspace(2, 11, 2)).astype(int),
#                 "min_samples_leaf"  : np.rint(np.linspace(1, 11, 2)).astype(int) }
#                 
#clf = RandomForestClassifier(oob_score=True) 
##grid_search = RandomizedSearchCV(clf, grid_test0, n_jobs=4, verbose=1000)
#grid_search = GridSearchCV(clf, grid_test0, n_jobs=4, verbose=1000)
#
#start = time()
#grid_search.fit(X,Y)
#print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#      % (time() - start, len(grid_search.grid_scores_)))      
#best = report(grid_search.grid_scores_)
#
#valid_depth(X,Y)
#valid_min_samples_leaf(X,Y)
#valid_max_features(X,Y)
