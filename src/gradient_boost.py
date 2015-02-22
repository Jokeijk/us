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

def final_run(X,Y,Xtest,n_est):
    clf = GradientBoostingClassifier(n_estimators=n_est,random_state=n_est)
    clf = clf.fit(X,Y)
    #np.savetxt('gb_oob_improve_{}'.format(n_est),clf.oob_score_)
    #np.savetxt('gb_train_score_{}'.format(n_est),clf.train_score_)
    Ytest=clf.predict(Xtest)
    output(Ytest,'gradient_boost_{}.csv'.format(n_est))

X,Y,Xtest=get_data()

#tic=time()
#final_run(X,Y,Xtest)
#toc=time()
#print "time {} second ".format(toc-tic)

#for n_est in range(100,1001,100):
#for n_est in range(1100,2001,100):
#for n_est in range(1050,1101,10):
#    tic=time()
#    clf = GradientBoostingClassifier(n_estimators=n_est)
#    score =  cross_validation.cross_val_score(clf,X,Y, cv=5, n_jobs=-1)
#    print "n_est=", n_est,"score=", score, " smean=", score.mean()
#    sys.stdout.flush()
#    toc=time()
#    print "time {} second ".format(toc-tic)

def main():
    Nest=int(sys.argv[1])
    print "Nest= ",Nest
    X,Y,Xtest=get_data()
    tic=time()
    final_run(X,Y,Xtest,Nest)
    toc=time()
    print "Nest {}: time {} second ".format(Nest, toc-tic)

if __name__ == '__main__':
    main()

#for depth in range(3,7):
#    for n_est in range(100,1001,100):
#        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
#                algorithm="SAMME",
#                n_estimators=n_est)
#        score =  cross_validation.cross_val_score(clf,X,Y, cv=10, n_jobs=-1)
#        print "depth=", depth,"n_est=", n_est,"score=", score, " smean=", score.mean()
