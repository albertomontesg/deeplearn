
num_layers = [3, 4, 5, 6, 7]
learning_rates = [.05, .10, .15, .20, .25, .30, .35, .40, .45]
activation_functions = ['tanh', 'sigmoid', 'logistic']
batch_sizes = [5, 10, 15, 20]
epochs = [5, 10, 15, 20]
array_experiments=[]

def generate_arguments_for_experiments():

	experiment = []

	set_num_layers(experiment)

	return activation_functions

def set_num_layers(experiment):
	for num in num_layers:
		exp = experiment[:]
		exp.append(num)
		set_learning_rate(exp)

def set_learning_rate(experiment):
	for lr in learning_rates:
		exp = experiment[:]
		exp.append(lr)
		set_act_func(exp)

def set_act_func(experiment):
	for act_func in activation_functions:
		exp = experiment[:]
		exp.append(act_func)
		set_batch_size(exp)

def set_batch_size(experiment):
	for batch_size in batch_sizes:
		exp = experiment[:]
		exp.append(batch_size)
		set_epcochs(exp)

def set_epcochs(experiment):
	for epoch in epochs:
		exp = experiment[:]
		exp.append(epoch)
		array_experiments.append(exp)

def print_experiments():
	for exp in array_experiments:
		print exp
	print len(array_experiments)

generate_arguments_for_experiments()
print_experiments()
