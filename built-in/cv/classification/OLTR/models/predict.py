import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import models.ResNet50Feature as resnet_feat
import models.DotProductClassifier as classifier
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import logging

def set_logging(console_level, file_level, file):
    logging.basicConfig(filename=file,level=file_level)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    logging.getLogger().addHandler(console)


class OLT_Model():
    def __init__(self, model_path='./logs/ImageNet_LT/stage1/final_model_checkpoint.pth', num_classes=4):
        self.model_path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.networks = {}

        self.net_feat = resnet_feat.create_model(test=True)
        self.net_feat = nn.DataParallel(self.net_feat).to(self.device)

        self.net_class = classifier.create_model(feat_dim=1024, num_classes=num_classes,test=True)
        self.net_class = nn.DataParallel(self.net_class).to(self.device)

        checkpoint = torch.load(self.model_path)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None

        #  feat_model classifier

        weights = model_state['feat_model']
        weights = {k: weights[k] for k in weights if k in self.net_feat.state_dict()}
        # model.load_state_dict(model_state[key])
        self.net_feat.load_state_dict(weights)
        self.net_feat.eval()

        #torch.save(self.net_feat, './net_feat.pth')

        weights = model_state['classifier']
        weights = {k: weights[k] for k in weights if k in self.net_class.state_dict()}
        # model.load_state_dict(model_state[key])
        self.net_class.load_state_dict(weights)
        self.net_class.eval()

        #torch.save(self.net_class, './net_class.pth')

        self.transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

        self.memory = {}
        self.memory['centroids'] = False
        self.memory['init_centroids'] = False

        # dummy_input1 = torch.randn(1, 3, 224, 224)
        # input_names = [ "input_0"]
        # output_names = [ "module" ]
        # # # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
        # torch.onnx.export(self.net_feat, dummy_input1, "C3AE_emotion.onnx", verbose=True, input_names=input_names, output_names=output_names)


    def process(self, image):        

        image_tensor = self.transform(image)

        image_tensor.unsqueeze_(0)

        image_tensor = image_tensor.to(self.device)
 
        features, feature_maps = self.net_feat(image_tensor)
        logits, direct_memory_feature = self.net_class(features, self.centroids)
        
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
        return preds.item(), probs.item()

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default='../data/clean_data_all',
    	help="root dir path (i.e., directory of data)")
    ap.add_argument("--txt",  default='../data/OLTR/UPPERSTYLE/test.txt',
    	help="path of test.txt")
    ap.add_argument("--outpath",  default='./logs/ImageNet_LT/upperstyle_stage1/',
    	help="path to load final_model and save log file")
    ap.add_argument("--name",  default='upperstyle',
    	help="attribute name")
    ap.add_argument("-c", "--cuda", default='0',
    	help="cuda No to use")
    ap.add_argument("--num_classes",  default=4,
    	help="num_classes")
    args = vars(ap.parse_args())
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    root = args['root']
    txt = args['txt']
    outpath = args['outpath']
    model_path = outpath+'final_model_checkpoint.pth'
    name = args['name']
    num_classes = int(args['num_classes'])
    handel = OLT_Model(model_path,num_classes)
    #set logging
    set_logging(logging.WARNING, logging.INFO,outpath+'/result_log')
    print(txt)
    logging.info(txt)
    reallist = []
    predlist = []
    with open(txt) as f:
        for line in f:
            if os.path.isfile(os.path.join(root, line.split()[0])):
                path = os.path.join(root, line.split()[0])
                label = int(line.split()[1])
                with open(path, 'rb') as pf:
                    image = Image.open(pf).convert('RGB')
                    preds, probs = handel.process(image)
                    print('{}: real class: {}  pred class:{}  source:{}'.format(line.split()[0],preds,label,probs))
                    reallist.append(label)
                    predlist.append(preds)
    X_cmat = confusion_matrix(reallist, predlist)
    X_pr = classification_report(reallist, predlist,digits=5)
    X_acc = accuracy_score(reallist, predlist)
    print(name+"_cmat:\n", X_cmat)
    logging.info(name+"_cmat:\n{}".format(X_cmat))
    print(name+"_pr:\n", X_pr)
    logging.info(name+"_pr:\n{}".format(X_pr))
    print(name+"_acc:", X_acc, '\n')
    logging.info(name+"_acc:{}'\n'".format(X_acc))
                        
