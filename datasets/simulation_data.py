
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.datasets.preprocessing import MakeUnitNorm

class SimulationData(object):
	def __init__(self, sim_path="./datasets/", save_path="data.pkl"):
		self.sim_path = sim_path
		self.save_path = save_path  
		self.input_file_name = 'simulation_input.txt'
		self.output_file_name = 'simulation_output.txt'
		
	def load_data(self):
		print "Starting loading data..."
		
		matrix_input = self.import_file(self.sim_path + self.input_file_name)
		matrix_output = self.import_file(self.sim_path + self.output_file_name)
		
		self.data = DenseDesignMatrix(X = matrix_input, y = matrix_output)
		print "Data load completed"
		return self.data
	
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
		array = np.array(matrix, dtype=np.float32)
		rows, colums = array.shape
		self.num_simulations = rows
		if colums > 100:
			self.input_values = colums
		else:
			self.output_values = colums
		return array
		
	def preprocessor(self):
		self.data.apply_preprocessor(preprocessor = MakeUnitNorm())
		self.preprocessor_for_output(self.data.y)

	def preprocessor_for_output(self, y):
		y_norm = np.sqrt(np.sum(y ** 2, axis=1))
		y /= y_norm[:, None]
		
	def save_data(self):
		serial.save(self.sim_path + self.save_path, self.data)
		print "Data saved in " + self.sim_path + self.save_path
		
	def get_matrix(self):
		return self.data
	
if __name__ == '__main__':
	sim = SimulationData()
	sim.load_data()
	sim.preprocessor()
	sim.save_data()
	print sim.data
	print sim.num_simulations
	print sim.input_values
	print sim.output_values
	x,y=sim.get_matrix().get_data()
	print x
	print y
	#print sim.get_matrix().get_batch_design(batch_size=100)