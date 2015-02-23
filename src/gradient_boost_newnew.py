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

from sklearn.ensemble import GradientBoostingClassifier


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

def dogridsearch(X,Y,param_space,clf):
    grid_search = GridSearchCV(clf,param_space,verbose=10l,cv=5,n_jobs=-1)
    start = time()
    grid_search.fit(X,Y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))      
    best = report(grid_search.grid_scores_)

def final_run(X,Y,Xtest):
    #clf = GradientBoostingClassifier(n_estimators=n_est,random_state=n_est)
    clf = GradientBoostingClassifier(
            n_estimators=2000,
            max_depth=3,
            max_features=None,
            learning_rate=0.08,
            random_state=0)
    clf = clf.fit(X,Y)
    #np.savetxt('gb_oob_improve_{}'.format(n_est),clf.oob_score_)
    #np.savetxt('gb_train_score_{}'.format(n_est),clf.train_score_)
    Ytest=clf.predict(Xtest)
    output(Ytest,'gradient_boost_newnew.csv')

X,Y,Xtest=get_data()

#tic=time()
#final_run(X,Y,Xtest)
#toc=time()
#print "time {} second ".format(toc-tic)

#for n_est in range(100,1001,100):
#for n_est in range(1100,2001,100):
def cross_val():
    clf = GradientBoostingClassifier()
    param_grid={
            "n_estimators":[2000],
            "max_depth":[2,3,4],
            "learning_rate":[0.05,0.1],
            "max_features":[22,100,300]
            }
    param_grid={
            "n_estimators":[3000],
            "max_depth":[3,4],
            "learning_rate":[0.1],
            "max_features":[22,300]
            }
    param_grid={
            "n_estimators":[2000],
            "max_depth":[3],
            "learning_rate":[0.1],
            "max_features":range(30,200,10)
            }
    param_grid={
            "n_estimators":range(1800,2500,100),
            "max_depth":[3],
            "learning_rate":[0.1]
            }
    param_grid={
            "n_estimators":[1800],
            "max_depth":[3],
            "learning_rate":[0.1,0.05,0.02]
            }
    param_grid={
            "n_estimators":[2000],
            "max_depth":[3,4,5],
            "learning_rate":[0.1],
            "max_features":[None,300]
            }
    param_grid={
            "n_estimators":[2000],
            "max_depth":[3,4],
            "learning_rate":[0.1,0.08],
            "max_features":[None,250,300,280,330]
            }
    dogridsearch(X,Y,param_grid,clf)

def main():
    tic=time()
    #cross_val()
    final_run(X,Y,Xtest)
    toc=time()
    print "time {} second ".format(toc-tic)

if __name__ == '__main__':
    #cross_val()
    main()
