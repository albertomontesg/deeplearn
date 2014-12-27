from datasets.simulation_data import SimulationData
import numpy as np
from numpy import mean, square
import matplotlib.pyplot as plt

s=SimulationData()
s.load_data()
tmp = s.split_train_test()

x_train = tmp[0].X
y_train = tmp[0].y
x_test = tmp[1].X
y_test = tmp[1].y
"""
Delete the columns of X (DenseDesignMatrix) which value is always 0
"""
X = x_train
Xt = x_test
for i in range(36):
	X = np.delete(X, i*36, 1)
	Xt = np.delete(Xt, i*36, 1)

"""
Compute:
		coeficients = inv(X'*X)*X'*y
"""
A = np.dot(np.transpose(X),X)
B = np.dot(np.linalg.inv(A),np.transpose(X))
coef = np.dot(B, y_train)

"""
Predict the from the test set of data
"""
y_pred = np.dot(Xt,coef)

"Show the Mean Square Error"
MSE = mean(square(y_test - y_pred))
print MSE

"""
Plot predicted data vs test data
"""
f, axarr = plt.subplots(8,8)
f = 0;
c = 0;
for i in range(len(y_test[0])):
	x = np.array([])
	y = np.array([])
	for j in range(len(y_test)):
		x = np.append(x, y_test[j][i])
		y = np.append(y, y_pred[j][i])
	axarr[f,c].plot(x, y, 'ro')
	c += 1
	if (c==8):
		c = 0
		f += 1
plt.show()