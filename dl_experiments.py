
import sys, os, stat

from experiment.ExperimentsArgumentsGenerator import generate_arguments_for_experiments as generator

# Load the template command files
command_template = open("template.cmd", 'r')
template_lines = command_template.readlines()

# Generate all the sets of arguments
first_id = 10000
arguments = generator()
num_jobs = len(arguments)

# Create outputs and errors directories
if !os.path.exists('./outputs'):
	os.mkdir('./outputs')
if !os.path.exists('./errors'):
	os.mkdir('./errors')

# Create all the command files
if os.path.exists('./commands'):
	os.system('rm -r ./commands')
os.mkdir("commands")
identifier = first_id
py_file = "SingleEncoderTraining.py"
for s in arguments:
	identifier += 1
	file_name = "command_%d.cmd" % identifier
	o=open("commands/" + file_name,'w')
	lines = template_lines[:]
	lines[2] = lines[2] % identifier
	lines[4] = lines[4] % identifier
	lines[5] = lines[5] % identifier
	lines[-1] = (lines[-1] % (py_file, identifier, s[0], s[1], s[2], s[3], s[4])) + '\n'
	o.writelines(lines)
	o.close()
	# Execute the job
	os.chmod('commands/' + file_name, stat.S_IRWXU)
	os.system('cat commands/' + file_name)
