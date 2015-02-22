# this files implements nearest neighbor method
# sklearn has two nearest neighbor methods for classification: kNN and radius-based NN
# clf1 is kNN, clf2 is radius-based kNN

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from dlinghu_functions import *


x_train, y_train, x_test = read_data()

knn1 = KNeighborsClassifier()
knn2 = RadiusNeighborsClassifier()
k_list = [5, 10, 20, 40, 60, 80, 120]
radius_list = np.logspace(0, 3, 7)
param_grid1 = dict(n_neighbors=k_list)
param_grid2 = dict(radius=radius_list)

clf1 = tune_parameters(knn1, x_train, y_train, param_grid1)
print 'done with kNN!'
clf2 = tune_parameters(knn2, x_train, y_train, param_grid2)
print 'done with radius-based NN!'

print_cv_scores(clf1, x_train, y_train)
print_cv_scores(clf2, x_train, y_train)

y_test_predict1 = clf1.predict(x_test)
# y_test_predict2 = clf2.predict(x_test)
# y_train_predict = clf1.predict(x_train)
output(y_test_predict1, 'kNN_001.csv')  # n_neighbors=20
# output(y_test_predict2, 'kNN_002.csv')  # radius=1.0

