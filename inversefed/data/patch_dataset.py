import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class PatchDataset(Dataset):
    def __init__(self, path_to_images, csv_path, fold, sample=0, transform=None):
        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv(csv_path)
        self.fold = fold
        self.df = self.df[self.df['fold'] == fold]
        # self.df = self.df[self.df['No Finding'] == 1]
        
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(frac=sample, random_state=42)
            print('subsample the training set with ratio %f' % sample)
        
        self.df = self.df.set_index('Image Index')
        self.PRED_LABEL = ['No Finding', 'Cardiomegaly', 'Edema', 
                            'Consolidation', 'Pneumonia', 'Atelectasis',
                            'Pneumothorax', 'Pleural Effusion']

        
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        filename = '{0:06d}'.format(self.df.index[idx])
        image = Image.open(os.path.join(self.path_to_images, filename+'.png'))
        image = image.convert('RGB')
        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
        
        if self.transform:
            image = self.transform(image)
        return image, label