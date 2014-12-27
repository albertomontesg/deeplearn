# Local imports
from datasets.simulation_data import SimulationData
# pylearn imports
from pylearn2.models.rbm import RBM
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.mlp import Default
from pylearn2.train import Train
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

class RBMTraining:
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

		tmp = self.sim_data.split_train_test()
		self.datasets = {'train' : tmp[0], 'test' : tmp[1]}

		self.num_simulations = self.sim_data.num_simulations
		self.input_values = self.sim_data.input_values
		self.output_values = self.sim_data.output_values

	def set_structure(self, num_layers = 4, shape = 'linear'):
		self.vis = self.input_values
		self.hid = self.output_values
		return [self.vis, self.hid]
		
		   
	def get_model(self):
		self.model = RBM(nvis=self.vis, nhid=self.hid, irange=.05)
		return self.model
	   
	def set_training_criteria(self, 
							learning_rate=0.05,
							batch_size=10, 
							max_epochs=10):
		
		self.training_alg = DefaultTrainingAlgorithm(batch_size = batch_size, 
													monitoring_dataset = self.datasets, 
													termination_criterion = EpochCounter(max_epochs))
	
	def set_extensions(self, extensions=None):
		self.extensions = None #[MonitorBasedSaveBest(channel_name='objective',
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
