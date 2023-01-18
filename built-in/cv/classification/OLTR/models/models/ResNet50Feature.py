from models.ResNetFeature import *
from utils import *
from resnet50_self import ResModel
        
def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, weightpath=None, test=False,*args):
    
    print('Loading Scratch ResNet 10 Feature Model.')
    resnet50 = ResModel()

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            resnet10 = init_weights(model=resnet50,
                                    weights_path='./logs/{}/{}/final_model_checkpoint.pth'.format(dataset,weightpath))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet50
