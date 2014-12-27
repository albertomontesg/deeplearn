
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.datasets.preprocessing import MakeUnitNorm

import time
import os

class SimulationData(object):
	def __init__(self, sim_path="./datasets/", save_path="./datasets/data.pkl"):
		self.sim_path = sim_path
		self.save_path = save_path  
		self.input_file_name = 'simulation_input.txt'
		self.output_file_name = 'simulation_output.txt'
		self.is_loaded = False
		
	def load_data(self):
		print ("Starting loading data...")
		starting = time.time()
		
		if(self.is_saved()):
			print "Loading file..."
			self.data = serial.load(self.sim_path + self.save_path)
			self.matrix_input = self.data.X.tolist()
			self.matrix_output = self.data.y.tolist()
		else:
			print "Importing..."
			self.matrix_input = self.import_file(self.sim_path + self.input_file_name)
			self.matrix_output = self.import_file(self.sim_path + self.output_file_name)
			
			self.data = DenseDesignMatrix(X = matrix_input, y = matrix_output)

		ending = time.time()
		print "Data load completed (%d seconds)" % (ending-starting)

		self.update_info()

		self.is_loaded = True
		return self.data
	
	def update_info(self):
		self.train_dataset = self.data
		self.num_simulations = len(self.data.y)
		self.input_values = len(self.data.X[0])
		self.output_values = len(self.data.y[0])

	def is_saved(self):
		return os.path.exists(self.sim_path + self.save_path)

	def is_loaded(self):
		return self.is_loaded

	def get_X_list(self):
		return self.matrix_input

	def get_y_list(self):
		return self.matrix_output

	def import_file(self, file):
		# Parse the content of the file specified 
		# and save the result into an array
		ofile=open(file, "r")
		lines=ofile.readlines()
		matrix = []
		c = 0
		r = 0
		matrix.append([])
		for i in range(len(lines)):
			if lines[i]=='\n':
				r = r + 1
				matrix.append([])
			else:
				values = lines[i].split('\t')
				try:
					for value in values:
						matrix[r].append(float(value))	
				except ValueError,e:
					#print "error",e,"on line",line
				 	c=c+1
	
		matrix.pop()
		return np.array(matrix, dtype=np.float32)

	def split_train_test(self, proportion = .8):
		# Number of training simulations
		tn = self.num_simulations * proportion

		X_train = self.data.X[:tn]
		y_train = self.data.y[:tn]

		X_test = self.data.X[tn:]
		y_test = self.data.y[tn:]

		self.train_dataset = DenseDesignMatrix(X = X_train, y = y_train)
		self.test_dataset = DenseDesignMatrix(X = X_test, y = y_test)

		return [self.train_dataset, self.test_dataset]
		
	def get_train_dataset(self):
		return self.train_dataset

	def get_test_dataset(self):
		return self.test_dataset

	def preprocessor(self, kind = 'uniform'):
		if(kind == 'uniform'):
			self.preprocessor_uniform()
		if(kind == 'logarithmic'):
			self.preprocessor_logarithmic()
		if(kind == 'unit_normalization'):
			self.preprocessor_unit_normalization()

	def preprocessor_uniform(self):
		minimum_x = np.amin(self.data.X)
		maximum_x = np.amax(self.data.X)
		mean_x = np.mean(self.data.X)
		self.data.X = (self.data.X - mean_x) / (2*(maximum_x - minimum_x))

		minimum_y = np.amin(self.data.y)
		maximum_y = np.amax(self.data.y)
		mean_y = np.mean(self.data.y)
		self.data.y = (self.data.y - mean_y) / (2*(maximum_y - minimum_y))

	def preprocessor_logarithmic(self):
		self.data.y = np.log10(self.data.y + 1)
		self.preprocessor_uniform()

	def preprocessor_unit_normalization(self):
		x_norm = np.sqrt(np.sum(self.data.X ** 2, axis=1))
		self.data.X /= x_norm[:, None]

		y_norm = np.sqrt(np.sum(self.data.y ** 2, axis=1))
		self.data.y /= y_norm[:, None]
		
	def reduce_output_single(self, position = 0):
		new_y = []
		for i in range(self.num_simulations):
			new_y.append([self.data.y[i][position]])
		y = np.array(new_y, dtype=np.float32)
		self.data = DenseDesignMatrix(X = self.data.X, y = y)

	def remove_input_zeros(self):
		a = int(np.sqrt(len(self.data.X[0])))
		for i in range(a):
			self.data.X = np.delete(self.data.X, i*a, 1)
		self.update_info()

	def save_data(self):
		serial.save(self.sim_path + self.save_path, self.data)
		print ("Data saved in ", self.sim_path, self.save_path)
		
	def get_matrix(self):
		return self.data
