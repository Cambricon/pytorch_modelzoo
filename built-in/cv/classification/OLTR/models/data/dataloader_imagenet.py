from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
# from keras.preprocessing.image import img_to_array

# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize((96,96)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 'train2': transforms.Compose([
    #     transforms.Resize((105,105)),
    #     transforms.RandomResizedCrop(96),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]),
    'val': transforms.Compose([
        # transforms.Resize((96,96)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.Resize((96,96)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None, aug=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        # self.aug = aug
        with open(txt) as f:
            for line in f:
                if os.path.isfile(os.path.join(root, line.split()[0])):
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        """
        if self.aug is not None:
            sample = cv2.imread(path)
            #sample = cv2.resize(sample, (96,96))
            # sample = img_to_array(sample)[np.newaxis,:,:,:]
            sample = sample[np.newaxis, :, :, :]
            labels = np.array(label)[np.newaxis]
            gen = self.aug.flow(sample, labels, batch_size=1, seed=42)
            sample,_ = gen.next()
            sample = transforms.ToPILImage()(sample[0].astype('uint8'))
        else:
            with open(path, 'rb') as f:
                sample = Image.open(f).convert('RGB')
        """
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path

# Load datasets
def load_data(data_root, data_dir, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True, aug = None, loadertype=''):
    #txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))
    #txt = '/home/mahongmin/dataset/plate_type/tain.txt'
    # if phase == 'train_plain':
    #     phase = 'train'
    txt = data_root + '/%s.txt'%(phase if phase != 'train_plain' else 'train')
    print('Loading data from %s' % (txt))

    """
    if phase == 'test':
        transform = data_transforms['test']
    elif phase == 'val':
        transform = data_transforms['val']
    else:
        transform = data_transforms[phase+loadertype]
    """
    if phase not in ['train', 'val']:
        transform = data_transforms['test']
    else:
        transform = data_transforms[phase]

    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_dir, txt, transform)

    if phase == 'test' and test_open:
        open_txt = data_root + '/open.txt'
        print('Testing with opensets from %s'%(open_txt))
        open_set_ = LT_Dataset(data_root + '_open', open_txt, transform)
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler.')
        print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
        
    
    
