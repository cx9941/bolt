from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *

if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)


    print('Pre-training begin...')
    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    print('Pre-training finished!')

    print('Evaluation begin...')
    manager_p.evaluation(args, data)
    print('Evaluation finished!')

    manager_p.save_results(args)
    
