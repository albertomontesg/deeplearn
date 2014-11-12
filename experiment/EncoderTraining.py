# Local imports
from datasets.simulation_data import SimulationData
# pylearn imports
from pylearn2.models.autoencoder import Autoencoder, DeepComposedAutoencoder
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

import sys

class EncoderTraining:
	def __init__(self, data_path="./datasets/", save_path="training.pkl", simulation_data = None, identifier = 0):
		self.id = identifier
		self.data_path = data_path
		self.save_path = save_path
		if simulation_data != None:
			self.sim_data = simulation_data
			self.save_data_loaded()
		else:
			self.sim_data = SimulationData(data_path)
			self.load_data()
		
	def load_data(self):
		self.sim_data.load_data()
		self.sim_data.preprocessor()
		self.save_data_loaded()

	def save_data_loaded(self):
		self.data_matrix = self.sim_data.get_matrix()
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
		
	def get_autoencoder(self, structure, act_function='sigmoid'):
		n_input, n_output = structure
		config = {
			'nvis': n_input,
			'nhid': n_output,
			'act_enc': act_function,
			'act_dec': act_function,
			"irange" : 0.05,
			}
		return Autoencoder(**config)
		
	def get_layers(self, act_function='tanh'):
		self.layers = []
		for pair in self.structure:
			self.layers.append(self.get_autoencoder(structure = pair, act_function=act_function))
		return self.layers
		   
	def get_model(self):
		self.model = DeepComposedAutoencoder(self.layers)
		return self.model
	   
	def set_training_criteria(self, 
							learning_rate=0.05, 
							cost=MeanSquaredReconstructionError(), 
							batch_size=10, 
							max_epochs=10):
		dataset = self.data_matrix
		self.training_alg = SGD(learning_rate = learning_rate, 
								cost = cost, 
								batch_size = batch_size, 
								monitoring_dataset = dataset, 
								termination_criterion = EpochCounter(max_epochs))
	
	def set_extensions(self, extensions=None):
		self.extensions = [MonitorBasedSaveBest(channel_name='objective',
												save_path = './training/training_monitor_best.pkl')]
		
	def set_attributes(self, attributes):
		self.attributes = attributes

	def define_training_experiment(self, save_freq = 10):
		self.experiment = Train(dataset=self.data_matrix, 
								model=self.model, 
								algorithm=self.training_alg, 
								save_path=self.save_path , 
								save_freq=save_freq, 
								allow_overwrite=True, 
								extensions=self.extensions)

	def train_experiment(self):
		self.experiment.main_loop()


if __name__ == '__main__':

    args = sys.argv
    print str(args)

    # learning_rate = args[1]
    # activation_function = args[2]
    # batch_size = args[3]
    # epochs = args[4]
    # identifier = args[0]

    # save_path = save_path+'training_encoder_%d.pkl' % (identifier)
    # experiment = EncoderTraining(data_path = self.data_path, 
    #                             save_path = save_path, 
    #                             simulation_data = self.sim_data,
    #                             identifier = identifier)

    # experiment.set_attributes(attributes)
    # # Set up the experiment
    # experiment.set_structure(num_layers = num_layers)
    # experiment.get_layers(encoder = activation_function)
    # experiment.get_model()
    # experiment.set_training_criteria(learning_rate = learning_rate,
    #                         batch_size = batch_size,
    #                         max_epochs = epochs)
    # experiment.set_extensions()
    # experiment.define_training_experiment()

    # experiment.train_experiment()
