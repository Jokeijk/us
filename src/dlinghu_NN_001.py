# this file tries neural network for classification
# it doesn't work yet
import numpy as np
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer, TanhLayer
from sklearn.metrics import accuracy_score
from dlinghu_functions import *

x_train, y_train, x_test = read_data()
train_data = ClassificationDataSet(x_train.shape[1], 1, nb_classes=2)
test_data = ClassificationDataSet(x_test.shape[1], 1, nb_classes=2)
# train_data = SupervisedDataSet(x_train.shape[1], 1)
# test_data = SupervisedDataSet(x_test.shape[1], 1)
train_data.setField('input', x_train)
train_data.setField('target', y_train.reshape(-1, 1))
test_data.setField('input', x_test)
test_data.setField('target', np.zeros((x_test.shape[0], 1)))
train_data._convertToOneOfMany()
test_data._convertToOneOfMany()
print train_data['input'], train_data['target'], test_data.indim, test_data.outdim
# for i in xrange(x_train.shape[0]):
# train_data.addSample(x_train[i, :], y_train[i])
# test_data.addSample(x_test[])

hidden_size = 20
fnn = buildNetwork(train_data.indim, hidden_size, train_data.outdim,
                   hiddenclass=SigmoidLayer,
                   outclass=SoftmaxLayer)
# fnn = buildNetwork(train_data.indim, hidden_size, train_data.outdim,
# hiddenclass=TanhLayer)
# trainer = BackpropTrainer(fnn, dataset=train_data,
#                           momentum=0.1, verbose=True,
#                           weightdecay=0.01, learningrate=0.01)
trainer = BackpropTrainer(fnn, train_data, verbose=True)
# trainer.trainEpochs(10)
trainer.trainUntilConvergence(verbose=True, validationProportion=0.15,
                              maxEpochs=1000, continueEpochs=10)
train_result = percentError(trainer.testOnClassData(),
                            train_data['class'])
print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % train_result

y_train_predict_raw = fnn.activateOnDataset(train_data)
print y_train_predict_raw

# find the best threshold
def find_nn_threshold(y_train_predict_raw, y_train):
    threshold_list = np.arange(start=y_train_predict_raw[:, 0].min(),
                               stop=y_train_predict_raw[:, 0].max(),
                               step=1e-5)
    threshold_best = 0
    accuracy_best = 0
    for threshold in threshold_list:
        y_train_predict = np.zeros(y_train.shape)
        y_train_predict[y_train_predict_raw[:, 0] < threshold] = 1
        tmp_score = accuracy_score(y_train_predict, y_train)
        # print '%s\t%s' % (threshold, tmp_score)
        print '{:>8} {:>8}'.format(*[threshold, tmp_score])
        if tmp_score > accuracy_best:
            accuracy_best = tmp_score
            threshold_best = threshold
    print 'best threshold = %s, with accuracy %s' % (threshold_best, accuracy_best)
    return threshold_best


threshold = find_nn_threshold(y_train_predict_raw, y_train)
y_train_predict = np.zeros(y_train.shape)
y_train_predict[y_train_predict_raw[:, 0] < threshold] = 1

# print np.float(fnn.activate(x_train[0, :]))
print y_train_predict.sum()
print accuracy_score(y_train_predict, y_train)

y_test_predict_raw = fnn.activateOnDataset(test_data)
y_test_predict = np.zeros((x_test.shape[0], 1))
y_test_predict[y_train_predict_raw[:, 0] < threshold] = 1
output(y_test_predict, 'NN_001.csv')