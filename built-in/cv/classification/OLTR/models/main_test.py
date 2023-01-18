import os
import argparse
import pprint
from data import dataloader_imagenet
from run_networks import model
import warnings
from utils import source_import

# ================
# LOAD CONFIGURATIONS

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/Imagenet_LT/Stage_1.py', type=str)
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--data_dir', type=str, default="")
args = parser.parse_args()

test_open = args.test_open
output_logits = args.output_logits

config = source_import(args.config).config
training_opt = config['training_opt']
# change
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from {}'.format(args.data_dir))
pprint.pprint(config)

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

print('Under testing phase, we load training data simply to calculate training data number for each class.')

data = {x: dataloader_imagenet.load_data(data_root='data/ImageNet_LT',
            data_dir=args.data_dir,
            dataset=dataset,
            phase=x,
            batch_size=training_opt['batch_size'],
            sampler_dic=None,
            test_open=test_open,
            num_workers=training_opt['num_workers'],
            shuffle=False) for x in ['train', 'test']}


training_model = model(config, data, test=True)
training_model.load_model()
training_model.eval(phase='test', openset=test_open)

if output_logits:
    training_model.output_logits(openset=test_open)

print('ALL COMPLETED.')
