
from experiment.MLPTraining import MLPTraining
from datasets.simulation_data import SimulationData
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
import theano
from numpy import mean, square
import numpy as np
from pylearn2.utils import serial
import matplotlib.pyplot as plt
import scipy

## Test MLP Training
identifier = 10002
num_layers = 1
learning_rate = 0.1
activation_function = 'linear'
batch_size = 10
epochs = 10
save_path = './training/training_linear_regressor_%d.pkl' % (identifier)

sim = SimulationData()
sim.load_data()
#sim.remove_input_zeros()
sim.preprocessor('uniform')

# Create the experiment
experiment = MLPTraining(save_path = save_path,
						simulation_data = sim,
						identifier = identifier,
						preprocessor = None)

print experiment.sim_data.data.X.shape

# Set up the experiment
experiment.set_structure(num_layers = num_layers)
experiment.get_layers(act_function=activation_function)
experiment.get_model(batch_size = batch_size)
experiment.set_training_criteria(batch_size = batch_size,
								learning_rate = learning_rate,
								max_epochs = epochs)
experiment.set_extensions(extensions = None) #[MonitorBasedLRAdjuster(dataset_name = 'train')])
experiment.define_training_experiment()

# Train the experiment
experiment.train_experiment()


[train, test] = sim.split_train_test()

experiment.predict(test=test)

# model = serial.load(save_path)
# X=model.get_input_space().make_theano_batch()
# Y=model.fprop(X)
# f=theano.function([X], Y)
# x_test = test.X
# y_test = test.y
# y_pred = f(x_test)

# MSE = mean(square(y_test - y_pred))
# print "MSE:", MSE
# var = mean(square(y_test))
# print "Var:", var


# f, axarr = plt.subplots(8,8)
# r = []
# s = []
# f = 0;
# c = 0;
# rx = np.linspace(0, .01, 1)
# for i in range(len(y_test[0])):
# 	x = np.array([])
# 	y = np.array([])
# 	for j in range(len(y_test)):
# 		x = np.append(x, y_test[j][i])
# 		y = np.append(y, y_pred[j][i])

# 	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
# 	ry = rx * slope + intercept
# 	r.append(r_value**2)
# 	s.append(slope)
# 	axarr[f,c].plot(x, y, 'ro')
# 	c += 1
# 	if (c==8):
# 		c = 0
# 		f += 1

# plt.figure()
# plt.plot(r)
# plt.figure()
# plt.hist(s)

# plt.show()