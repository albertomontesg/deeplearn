import sys
from experiment.EncoderTraining  import EncoderTraining

if __name__ == '__main__':

	# Take the arguments and print them in console to identifie the encoder
	args = sys.argv
	print "# Identifier: " + args[1]
	print "# Number of layers: " + args[2]
	print "# Learning Rate: " + args[3]
	print "# Activation Function: " + args[4]
	print "# Batch Size: " + args[5]
	print "# Epochs: " + args[6]

	# Parse the arguments
	identifier = int(args[1])
	num_layers = int(args[2])
	learning_rate = float(args[3])
	activation_function = args[4]
	batch_size = int(args[5])
	epochs = int(args[6])
	save_path = './training/training_encoder_%d.pkl' % (identifier)
	# Create the experiment
	experiment = EncoderTraining(save_path = save_path,
								identifier = identifier)

	experiment.set_attributes(args[2:])
	# Set up the experiment
	experiment.set_structure(num_layers = num_layers)
	experiment.get_layers(act_function = activation_function)
	experiment.get_model()
	experiment.set_training_criteria(learning_rate = learning_rate,
	                        batch_size = batch_size,
	                        max_epochs = epochs)
	experiment.set_extensions()
	experiment.define_training_experiment()

	# Train the experiment
	experiment.train_experiment()
