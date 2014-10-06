
from pylearn2.models.rbm import RBM
from pylearn2.models.autoencoder import Autoencoder, DeepComposedAutoencoder
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from import_data import SimulationData

ninput = 1296
noutput = 61


def train_model():
    global ninput, noutput
    simdata = SimulationData(sim_path="../../javaDataCenter/generarDadesV1/CA_SDN_topo1/")
    simdata.load_data()
    simdata.preprocessor() 
    dataset = simdata.get_matrix()
    
    structure = get_structure()
    layers = []
    for pair in structure:
        layers.append(get_autoencoder(pair))
      
    model = DeepComposedAutoencoder(layers)
    training_alg = SGD(learning_rate=1e-3, cost=MeanSquaredReconstructionError(), batch_size=1296, monitoring_dataset=dataset , termination_criterion=EpochCounter(max_epochs=50))
    extensions = [MonitorBasedLRAdjuster()]
    experiment = Train(dataset=dataset , model=model, algorithm=training_alg, save_path='training2.pkl' , save_freq=10, allow_overwrite=True, extensions=extensions)
    experiment.main_loop()
    
def get_rbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange" : 0.05,
        "init_bias_hid" : 0.0,
        "init_bias_vis" : 0.0,
        }

    return RBM(**config)
    
def get_autoencoder(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        'act_enc': 'sigmoid',
        'act_dec': None,
        "irange" : 0.05,
        }
    return Autoencoder(**config)
    
def get_structure():
    global ninput, noutput
    return [[1296, 884],[884, 473],[473, 61]]
    
if __name__ == '__main__':
    train_model()