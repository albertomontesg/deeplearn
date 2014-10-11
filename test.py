
from experiment.Experiment import Experiment, EncoderTraining

ex = Experiment()
ex.load_data()
ex.set_experiments()
ex.run_experiments()

# et = EncoderTraining()
# et.set_structure()
# et.get_layers()
# et.get_model()
# et.set_training_criteria()
# et.set_extensions()
# et.define_training_experiment()
# et.train_experiment()