
from pylearn2.train import Train
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.energy_functions.rbm_energy import grbm_type_1
from pylearn2.training_algorithms.sgd import SGD, MonitorBasedLRAdjuster
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.corruption import GaussianCorruptor
from pylearn2.termination_criteria import MonitorBased

def train_example():
    model = GaussianBinaryRBM(nvis=1296, nhid=61, irange=0.5,
                            energy_function_class=grbm_type_1(), learn_sigma=True,
                            init_sigma=.4, init_bias_hid=2., mean_vis=False,
                            sigma_lr_scale=1e-3)
    cost = SMD(corruptor=GaussianCorruptor(stdev=0.4))
    algorithm = SGD(learning_rate=.1, batch_size=5, monitoring_batches=20,
                    monitoring_dataset='data.pkl', cost=cost,
                    termination_criterion=MonitorBased(prop_decrease=0.01, N=1))
    train = Train(dataset='data.pkl',model=model, algoritm=algorithm, extensions=[MonitorBasedLRAdjuster()])
    train.main_loop()

if __name__ == '__main__':
    train_example()