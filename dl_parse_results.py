
from experiment.ExperimentsArgumentsGenerator import generate_arguments_for_experiments as generator
import re

# Number of simulations
num_simulations = len(generator())

results_file = open('./results.csv', 'w')


results_file.writelines(["Id;num_layers;lr;act_function;batch_size;epochs;objective"])
identifier = 10000
for i in range(num_simulations):
	identifier += 1
	file_name = "EncoderTraining_%d.out" % identifier
	o = open('outputs/' + file_name, 'r')
	lines = o.readlines()
	o.close()
	ident = lines[0].split(': ')[1].split('\n')[0]
	num_layers = lines[1].split(': ')[1].split('\n')[0]
	lr = lines[2].split(': ')[1].split('\n')[0]
	act_function = lines[3].split(': ')[1].split('\n')[0]
	batch_size = lines[4].split(': ')[1].split('\n')[0]
	epochs = lines[5].split(': ')[1].split('\n')[0]
	#print(ident, num_layers, lr, act_function, batch_size, epochs)

	# Search the last value of the objective funtion
	objective = ''
	for l in lines:
		l = l.rstrip()
		if re.search('objective:', l):
			objective = l.split('objective: ')[1].split('\n')[0]
	#print objective

	single_result = ident + ';' + num_layers + ';' + lr + ';' + act_function + ';' + batch_size + ';' + epochs + ';' + objective+ '\n'
	results_file.writelines([single_result])

results_file.close()