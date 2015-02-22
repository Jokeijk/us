import numpy as np
import pandas as pd
import sys

def output(Ytest,filename):
    out = pd.DataFrame({'Id':1+np.arange(Ytest.size),'Prediction':Ytest.astype(int)})
    out.to_csv(filename,index=False)

files=sys.argv[1:]



total=len(files)
assert(total>1)
vote=0

for f in files:
    a=np.loadtxt(f,skiprows=1,delimiter=',')
    vote=a[:,1]+vote

print "not agreeed on {} points ", np.sum(np.logical_and(vote > 0, vote<total))

per = vote > total/2+1
vote = vote*0
vote[per]=1

output(vote, "gradientboost_vote.csv")
