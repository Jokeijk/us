import numpy as np
import pandas as pd
from sklearn import tree

def output(Ytest,filename):
    out = pd.DataFrame({'Id':1+np.arange(Ytest.size),'Prediction':Ytest})
    out.to_csv(filename,index=False)
        
    

train_file = '../data/kaggle_train_tf_idf.csv'
test_file  = '../data/kaggle_test_tf_idf.csv'

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)


clf = tree.DecisionTreeClassifier()

X=df_train.take(range(1,df_train.shape[1]-1), axis=1)
Y=df_train.take([df_train.shape[1]-1],axis=1)

Xtest=df_test.take(range(1,df_train.shape[1]-1), axis=1)

clf.fit(X,Y)

Ytest=clf.predict(Xtest)

output(Ytest,'try_001.csv')
