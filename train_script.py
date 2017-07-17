#          lr output iter gpu console
# train.sh 0.01 300 5000 0.1 3 4 0.7 100 0 (1)
import numpy as np
import scipy.io as sio
import warnings
import dataset
import net as model
import sys
from pprint import pprint

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

### Define input arguments
lr = float(sys.argv[1])
output_dim = int(sys.argv[2])
iter_num = int(sys.argv[3])
cq_lambda = float(sys.argv[4])
subspace_num = int(sys.argv[5])
gl_loss = sys.argv[6]
_dataset = sys.argv[7]
gpu = sys.argv[8]
graph_laplacian_temperature = float(sys.argv[9])
graph_laplacian_k = int(sys.argv[10])
graph_laplacian_lambda = float(sys.argv[11])
log_dir = sys.argv[12]
label_ratio = 0.1 # float(sys.argv[12])

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 81}

config = {
    'device': '/gpu:' + gpu,
    'gpu_usage': 11,#G
    'max_iter': iter_num,
    'batch_size': 256, # changed
    'moving_average_decay': 0.9999,      # The decay to use for the moving average.
    'decay_step': 500,                   # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.5,   # Learning rate decay factor.
    'learning_rate': lr,                 # Initial learning rate img.
    'console_log': False,

    'output_dim': output_dim,

    'R': 5000,
    'model_weights': 'pretrained_model/reference_pretrain.npy',

    'img_model': 'alexnet',
    'stage': 'train',
    'loss_type': 'cos_softmargin_multi_label',

    'margin_param': 0.7,
    'wordvec_dict': "./data/{dataset}/{dataset}_wordvec.txt".format(dataset=_dataset),
    'part_ids_dict': "./data/{dataset}/ids.txt".format(dataset=_dataset),
    'label_ratio': label_ratio,
    'partlabel': label_ratio,

    # only finetune last layer
    'finetune_all': True,

    ## CQ params
    'max_iter_update_b': 3,
    'max_iter_update_Cb': 1,
    'cq_lambda': cq_lambda,
    'code_batch_size': 500,
    'n_subspace': subspace_num,
    'n_subcenter': 256,

    'graph_laplacian_temperature': graph_laplacian_temperature,
    'graph_laplacian_k': graph_laplacian_k,
    'graph_laplacian_lambda': graph_laplacian_lambda,
    'graph_laplacian_loss': gl_loss,

    'label_dim': label_dims[_dataset],
    'img_tr': "./data/{}/train.txt".format(_dataset),
    'img_te': "./data/{}/test.txt".format(_dataset),
    'img_db': "./data/{}/database.txt".format(_dataset),
    'save_dir': "./models/",
    'log_dir': log_dir,

    'dataset': _dataset
}

pprint(config)

import time
t = time.time()
train_img = dataset.import_train(config)
print time.time() - t

model_dq = model.train(train_img, config)

