# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
from fc_net import *
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array,rel_error
from solver import Solver
from get_data import Get_data
from decimal import Decimal

path = '/home/su/code/sEMG/3.14sEMG/code/data/features'
fileName = 'feature1.txt'
data = Get_data(path, fileName)
data.get_data()
chan1_features = data.features['chan1']
chan2_features = data.features['chan2']
chan3_features = data.features['chan3']
chan4_features = data.features['chan4']

X_train_list = []
for i in xrange(len(chan1_features)):
  temp = []
  temp.extend(chan1_features[i])
  temp.extend(chan2_features[i])
  temp.extend(chan3_features[i])
  temp.extend(chan4_features[i])

  X_train_list.append(temp)
X_train_temp = np.array(X_train_list)
X_train = X_train_temp.reshape((len(chan1_features),-1))

y_train = np.zeros((len(chan1_features),1),dtype='int')


#inputdata
data = {
  'X_train':X_train[16:],   # training data
  'y_train':y_train[16:],   # training labels
  'X_val':X_train[0:16],   # validation data
  'y_val':y_train[0:16]   # validation labels
}

##########check data's shape#################
for k, v in data.iteritems():
  print '%s: ' % k, v.shape
######################################

learning_rate = 1e-5 #3.1e-4
weight_scale=5e-2 #2.5e-2 #1e-5

solvers = {}
for update_rule in ['sgd']:#, 'sgd_momentum']:
  print 'running with ', update_rule
  #three-layer network with 10 units in each hidden layer
  model = FullyConnectedNet([10, 10],
                            input_dim=4*19, #肌电信号四个通道，每个通道19个特征值
                            num_classes=8,  #8个动作
                            #dropout=0.25, use_batchnorm=True, 
                            reg=1e-2,
                            weight_scale=weight_scale, 
                            dtype=np.float64
                           )


  solver = Solver(model, data,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': learning_rate,
                  },
                  #lr_decay=0.95,
                  print_every=10, num_epochs=10, batch_size=5,
                  verbose=True
                 )
  solvers[update_rule] = solver
  solver.train()




####################可视化#####################
plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

for update_rule, solver in solvers.iteritems():
  plt.subplot(3, 1, 1)
  plt.plot(solver.loss_history, '-o', label=update_rule)
  
  plt.subplot(3, 1, 2)
  plt.plot(solver.train_acc_history, '-o', label=update_rule)

  plt.subplot(3, 1, 3)
  plt.plot(solver.val_acc_history, '-o', label=update_rule)
  
for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()
