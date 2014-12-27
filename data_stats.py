
import theano
from numpy import mean, square
import numpy as np
from pylearn2.utils import serial
from datasets.simulation_data import SimulationData
import matplotlib.pyplot as plt
import scipy

sim = SimulationData()
sim.load_data()
sim.preprocessor()
[train, test] = sim.split_train_test()

dataset = sim.data

x = np.array([])
y = np.array([])
plt.figure()

plt.hist(sim.data.X[0])
plt.show()
# for i in range(len(sim.data.X)):
# 	plt.hist(sim.data.X[i])
# 	x = np.append(x, sim.data.X[i])
# 	y = np.append(y, sim.data.X[i])


# plt.hist(x)
# plt.show()

# plt.figure()
# plt.hist(y)
# plt.show()