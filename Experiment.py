from EncoderTraining import EncoderTraining
from datasets.simulation_data import SimulationData
from training.ExperimentsArgumentsGenerator import generate_arguments_for_experiments


class Experiment(object):
	"""docstring for Experiment"""
	def __init__(self, data_path='./datasats/', save_path='./'):
		super(Experiment, self).__init__()
		self.data_path = data_path
		self.save_path = save_path

		# Save the different experiments into an array
		self.experiments = []

		self.sim_data = SimulationData(data_path)

	def load_data(self):
		"""
		Load data and store it once to be accessible for each experiment
		that is going to be run
		"""
		self.sim_data.load_data()
		self.sim_data.preprocessor()
		self.sim_data.save_data()

	def set_experiments(self, attributes=None):
		if attributes == None:
			self.experiments_arguments = generate_arguments_for_experiments()
		else:
			self.experiments_arguments = attributes

		# for arg in self.experiments_arguments:
		# 	set_single_experiment(attributes = arg)
		self.set_single_experiment(attributes=self.experiments_arguments[0])
		
	def set_single_experiment(self, 
							num_layers=4,
							learning_rate=0.05,
							activation_function='tanh',
							batch_size = 10,
							epochs = 10,
							attributes = None):
		"""
		Possible values for the inputs:
		- num_layers = 3 to 7
		- learning_rate = from 0.05 to 0.45 with jumps of 0.05
		- activation_function = tanh, logistic, sigmoideal
		- batch_size = 5 to 20 with jumps of 5
		- epochs = 5 to 20 with jumps of 5 epochs
		"""
		if(num_layers == None): num_layers = args[0]
		if(learning_rate == None): learning_rate = args[1]
		if(activation_function == None): activation_function = args[2]
		if(batch_size == None): batch_size = args[3]
		if(epochs == None): epochs = args[4]

		save_path = self.save_path+'%d.pkl' % (len(self.experiments))
		experiment = EncoderTraining(data_path = self.data_path, 
									save_path = save_path, 
									simulation_data = self.sim_data)

		# Set up the experiment
		experiment.set_structure(num_layers = num_layers)
		experiment.get_layers(encoder = activation_function)
		experiment.get_model()
		experiment.set_training_criteria(learning_rate = learning_rate,
								batch_size = batch_size,
								max_epochs = epochs)
		experiment.set_extensions()
		experiment.define_training_experiment()

		self.experiments.append(experiment)

	def run_experiments(self):
		i = 0
		for exp in self.experiments:
			print "Running experiment %d:" % i
			i = i + 1
			exp.train_experiment()

if __name__ == '__main__':
	ex = Experiment()
	ex.load_data()
	ex.set_experiments()
	ex.run_experiments()
