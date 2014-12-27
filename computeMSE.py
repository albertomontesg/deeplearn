
import theano
from numpy import mean, square
import numpy as np
from pylearn2.utils import serial
from datasets.simulation_data import SimulationData
import matplotlib.pyplot as plt
import scipy

sim = SimulationData()
sim.load_data()

[train, test] = sim.split_train_test()

model = serial.load('./training/training_encoder_10001.pkl')
X=model.get_input_space().make_theano_batch()
Y=model.fprop(X)
f=theano.function([X], Y)
x_test = test.X
y_test = test.y
y_pred = f(x_test)

MSE = mean(square(y_test - y_pred))
print "MSE:", MSE
var = mean(square(y_test))
print "Var:", var


f, axarr = plt.subplots(8,8)
r = []
f = 0;
c = 0;
rx = np.linspace(0, .01, 1)
for i in range(len(y_test[0])):
	x = np.array([])
	y = np.array([])
	for j in range(len(y_test)):
		x = np.append(x, y_test[j][i])
		y = np.append(y, y_pred[j][i])

	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	ry = rx * slope + intercept
	r.append(r_value**2)
	axarr[f,c].plot(x, y, 'ro', rx, ry, '-')
	c += 1
	if (c==8):
		c = 0
		f += 1

plt.figure()
plt.plot(r)
plt.show()