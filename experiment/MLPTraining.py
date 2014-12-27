# Local imports
from datasets.simulation_data import SimulationData
# pylearn imports
from pylearn2.models.mlp import MLP, Linear, Sigmoid, Tanh
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.mlp import Default
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.utils import serial
import matplotlib.pyplot as plt
import theano
import scipy
import numpy as np

class MLPTraining:
	def __init__(self, data_path="./datasets/", save_path="training.pkl", simulation_data = None, identifier = 0, preprocessor='uniform'):
		self.id = identifier
		self.data_path = data_path
		self.save_path = save_path
		if simulation_data != None:
			self.sim_data = simulation_data
		else:
			self.sim_data = SimulationData(data_path)
		if not self.sim_data.is_loaded:
			self.sim_data.load_data()

		self.sim_data.preprocessor(kind = preprocessor)

		tmp = self.sim_data.split_train_test()
		self.datasets = {'train' : tmp[0], 'test' : tmp[1]}

		self.num_simulations = self.sim_data.num_simulations
		self.input_values = self.sim_data.input_values
		self.output_values = self.sim_data.output_values

	def set_structure(self, num_layers = 4, shape = 'linear'):
		structure = []

		lower_number = self.input_values
		for i in range(num_layers):
			upper_number = lower_number
			lower_number = self.input_values-(i+1)*(self.input_values-self.output_values)/num_layers
			structure.append([upper_number, lower_number])
		
		self.structure = structure
		return structure
		
	def get_structure(self):
		return self.structure
		
	def get_Linear_Layer(self, structure, i = 0):
		n_input, n_output = structure
		config = {
			'dim': n_output,
			'layer_name': ("l%d" % i),
			'irange': .5,
			'use_abs_loss': False,
			'use_bias': False,
			}
		return Linear(**config)

	def get_Sigmoid_Layer(self, structure, i = 0):
		n_input, n_output = structure
		config = {
			'dim': n_output,
			'layer_name': ("s%d" % i),
			'irange' : 0.05,
			}
		return Sigmoid(**config)

	def get_Tanh_Layer(self, structure, i = 0):
		n_input, n_output = structure
		config = {
			'dim': n_output,
			'layer_name': ("t%d" % i),
			'irange' : 0.05,
			}
		return Tanh(**config)
		
	def get_layers(self, act_function='linear'):
		self.layers = []
		i = 0
		for pair in self.structure:
			i += 1
			if(act_function == 'linear'):
				self.layers.append(self.get_Linear_Layer(structure = pair, i = i))
			if(act_function == 'sigmoid'):
				self.layers.append(self.get_Sigmoid_Layer(structure = pair, i = i))
			if(act_function == 'tanh'):
				self.layers.append(self.get_Tanh_Layer(structure = pair, i = i))
		return self.layers
		   
	def get_model(self, batch_size):
		vis = self.structure[0][0]
		self.model = MLP(layers = self.layers, nvis = vis, batch_size = batch_size, layer_name = None)
		return self.model
	   
	def set_training_criteria(self, 
							learning_rate=0.05, 
							cost=Default(), 
							batch_size=10, 
							max_epochs=10):
		
		self.training_alg = SGD(learning_rate = learning_rate, 
								cost = cost, 
								batch_size = batch_size, 
								monitoring_dataset = self.datasets, 
								termination_criterion = EpochCounter(max_epochs))
	
	def set_extensions(self, extensions):
		self.extensions = extensions #[MonitorBasedSaveBest(channel_name='objective',
												#save_path = './training/training_monitor_best.pkl')]
		
	def set_attributes(self, attributes):
		self.attributes = attributes

	def define_training_experiment(self, save_freq = 10):
		self.experiment = Train(dataset=self.datasets['train'], 
								model=self.model, 
								algorithm=self.training_alg, 
								save_path=self.save_path , 
								save_freq=save_freq, 
								allow_overwrite=True, 
								extensions=self.extensions)

	def train_experiment(self):
		self.experiment.main_loop()
		self.save_model()

	def save_model(self):
		self.model = serial.load(self.save_path)
		
	def predict(self, test=None, X=None, y=None):
		if test != None:
			x_test = test.X
			y_test = test.y
		else:
			x_test = X
			y_test = y

		X=self.model.get_input_space().make_theano_batch()
		Y=self.model.fprop(X)
		f=theano.function([X], Y)

		y_pred = f(x_test)

		if y_test != None:
			MSE = np.mean(np.square(y_test - y_pred))
			print "MSE:", MSE
			var = np.mean(np.square(y_test))
			print "Var:", var
			self.plot_prediction(y_test, y_pred)
		else:
			return y_pred

	def plot_prediction(self, y_test, y_pred):
		m = int(np.sqrt(self.output_values)) + 1
		f, axarr = plt.subplots(m,m)

		r = []
		s = []
		f = 0;
		c = 0;
		for i in range(self.output_values):
			x = np.array([])
			y = np.array([])
			for j in range(len(y_test)):
				x = np.append(x, y_test[j][i])
				y = np.append(y, y_pred[j][i])

			slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
			r.append(r_value**2)
			axarr[f,c].plot(x, y, 'ro')
			c += 1
			if (c==m):
				c = 0
				f += 1

		plt.show()
