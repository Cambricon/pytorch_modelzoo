import os
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
from utils import source_import

# ================
# LOAD CONFIGURATIONS
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='brand_mid_stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--attri', default='brand_mid')
parser.add_argument('--cuda', default='0')
parser.add_argument('--loadertype', default='')
parser.add_argument("--data_path",  default='../data/vehicle_brand',
	help="input dataset path (i.e., directory of dataset)")
args = parser.parse_args()
loadertype = args.loadertype
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
data_root = {'ImageNet': 'data/ImageNet_LT/'}
data_dir = args.data_path
test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
configpath = 'config/ImageNet_LT/'+args.config
config = source_import(configpath).config
training_opt = config['training_opt']
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']

if loadertype != '':
    training_opt['log_dir'] += ('_train'+loadertype)
    if 'weightpath' in config['networks']['feat_model']['params']:
        config['networks']['feat_model']['params']['weightpath'] += ('_train'+loadertype)
        config['networks']['classifier']['params']['weightpath'] += ('_train'+loadertype)

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(), 
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None

    data = {
        x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], 
        data_dir=data_dir, 
        dataset=dataset, 
        phase=x, 
        batch_size=training_opt['batch_size'],
        sampler_dic=sampler_dic,
        num_workers=training_opt['num_workers'],
        loadertype=loadertype) for x in (['train', 'val', 'train_plain'] if relatin_opt['init_centroids'] else ['train', 'val'])}

    training_model = model(config, data, test=False)

    training_model.train()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False,loadertype=loadertype)
            for x in ['train', 'test']}

    
    training_model = model(config, data, test=True)
    training_model.load_model()
    training_model.eval(phase='test', openset=test_open)
    
    if output_logits:
        training_model.output_logits(openset=test_open)
        
print('ALL COMPLETED.')
