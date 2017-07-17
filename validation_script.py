#                   model_path                gpu console
# validation.sh lr_0.01_output_300_iter_1000.npy 0 (1)
import numpy as np
import scipy.io as sio
import warnings
import dataset
import net_val as model
import sys

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

model_weight = sys.argv[1]
gpu = sys.argv[2]
_dataset = sys.argv[3]
subspace_num = int(model_weight.split('_')[6])
label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 81}

config = {
    'device': '/gpu:' + gpu,
    'gpu_usage': 11,#G
    'max_iter': 5000,
    'batch_size': 100,
    'moving_average_decay': 0.9999,      # The decay to use for the moving average.
    'decay_step': 500,          # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.1,   # Learning rate decay factor.
    'learning_rate': 0.001,       # Initial learning rate img.

    'output_dim': 300,

    'R': 5000,
    # trained model for validation
    'model_weights': model_weight,

    'img_model': 'alexnet',
    'stage': 'validation',
    'loss_type': 'cos_softmargin_multi_label',

    'margin_param': 0.7,
    'wordvec_dict': "./data/{dataset}/{dataset}_wordvec.txt".format(dataset=_dataset),
    'part_ids_dict': "",
    'partlabel': "",

    'graph_laplacian_temperature': 0,
    'graph_laplacian_k': 5,
    'graph_laplacian_lambda': 0.001,

    # only finetune last layer
    'finetune_all': False,

    'max_iter_update_b': 3,
    'max_iter_update_Cb': 1,
    'cq_lambda': 0.1,
    'code_batch_size': 50 * 14,
    'n_subspace': subspace_num,
    'n_subcenter': 256,

    'label_dim': label_dims[_dataset],
    'img_tr': "./data/{}/train_semi.txt".format(_dataset),
    'img_te': "./data/{}/test.txt".format(_dataset),
    'img_db': "./data/{}/database.txt".format(_dataset),
    'save_dir': "./models/",
}

query_img, database_img = dataset.import_validation(config)

model.validation(database_img, query_img, config)
