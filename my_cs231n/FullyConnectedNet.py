import time
import numpy as np
import matplotlib.pyplot as plt
from fc_net import *
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from solver import Solver

learning_rate = 8e-3 #3.1e-4
weight_scale=1e-2 #2.5e-2 #1e-5
#three-layer network with 100 units in each hidden layer
model = FullyConnectedNet([100, 100],
                          input_dim=4*19, #肌电信号四个通道，每个通道19个特征值
                          num_classes=8,  #8个动作
                          dropout=0.25, use_batchnorm=True, reg=1e-2,
                          weight_scale=weight_scale, 
                          dtype=np.float64
                         )

solver = Solver(model, [inputdata],
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                },
                lr_decay=0.95,
                print_every=10, num_epochs=20, batch_size=25,
                verbose=True
               )
solver.train()

#可视化
#v1
#loss value
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()
#accuracy
plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

##v2
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
  plt.plot(solver.loss_history, 'o', label=update_rule)
  
  plt.subplot(3, 1, 2)
  plt.plot(solver.train_acc_history, '-o', label=update_rule)

  plt.subplot(3, 1, 3)
  plt.plot(solver.val_acc_history, '-o', label=update_rule)
  
for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()
