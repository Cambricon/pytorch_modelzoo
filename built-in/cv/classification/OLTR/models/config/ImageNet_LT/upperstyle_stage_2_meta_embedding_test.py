# Testing configurations
config = {}

training_opt = {}
training_opt['dataset'] = 'ImageNet_LT'
training_opt['log_dir'] = './logs/ImageNet_LT/upperstyle_meta_embedding'
training_opt['num_classes'] = 4
training_opt['batch_size'] = 32
training_opt['num_workers'] = 0
training_opt['num_epochs'] = 2#30
training_opt['display_step'] = 10
training_opt['feature_dim'] = 1024
training_opt['open_threshold'] = 0.1
training_opt['sampler'] = {'type': 'ClassAwareSampler', 'def_file': './data/ClassAwareSampler.py',
                           'num_samples_cls': 4}
training_opt['scheduler_params'] = {'step_size': 20, 'gamma': 0.1}
config['training_opt'] = training_opt

networks = {}
feature_param = {'use_modulatedatt': True, 'use_fc': True, 'dropout': None,
                 'stage1_weights': True, 'dataset': training_opt['dataset'],'weightpath':'upperstyle_stage1'}
feature_optim_param = {'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 0.0005}
networks['feat_model'] = {'def_file': './models/ResNet50Feature.py',
                          'params': feature_param,
                          'optim_params': feature_optim_param,
                          'fix': False}
classifier_param = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes'],
                    'stage1_weights': True, 'dataset': training_opt['dataset'],'weightpath':'upperstyle_stage1'}
classifier_optim_param = {'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 0.0005}
networks['classifier'] = {'def_file': './models/MetaEmbeddingClassifier.py',
                          'params': classifier_param,
                          'optim_params': classifier_optim_param}
config['networks'] = networks

criterions = {}
perf_loss_param = {}
criterions['PerformanceLoss'] = {'def_file': './loss/SoftmaxLoss.py', 'loss_params': perf_loss_param,
                                 'optim_params': None, 'weight': 1.0}
feat_loss_param = {'feat_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes']}
feat_loss_optim_param = {'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 0.0005}
criterions['FeatureLoss'] = {'def_file': './loss/DiscCentroidsLoss.py', 'loss_params': feat_loss_param,
                             'optim_params': feat_loss_optim_param, 'weight': 0.01}
config['criterions'] = criterions

memory = {}
memory['centroids'] = True
memory['init_centroids'] = True
config['memory'] = memory
