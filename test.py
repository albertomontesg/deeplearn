
from experiment.Experiment import Experiment, EncoderTraining
import experiment.train_example as te
import datasets.simulation_data as sd

data = sd.SimulationData()
data.load_data()
data.preprocessor()
dataset = data.get_matrix()
te.train_example(dataset=dataset)


# ex = Experiment()
# ex.load_data()
# ex.set_experiments()
# ex.run_experiments()

# et = EncoderTraining()
# et.set_structure()
# et.get_layers()
# et.get_model()
# et.set_training_criteria()
# et.set_extensions()
# et.define_training_experiment()
# et.train_experiment()