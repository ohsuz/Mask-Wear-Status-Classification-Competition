import pandas as pd
import numpy as np
import torch
import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from .config import Config as conf


class PreprocessedDataset(Dataset):
    def __init__(self, train_csv_path, crop, opt: str):
        self.train_df = None
        self.preprocessing(train_csv_path, crop)
        self.images = self.get_images()
        self.labels = self.get_labels(opt)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.train_df)

    def preprocessing(self, train_csv_path, crop): # crop: whether getting cropped images
        df = pd.read_csv(train_csv_path)
        
        df['gender_age'] = df.apply(lambda x: self.convert_gender_age(x.gender, int(x.age)), axis=1)

        skf = StratifiedKFold(n_splits=conf.n_fold, shuffle=True)
        df.loc[:, 'Fold'] = 0
        for fold_num, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df.gender_age.values)):
            df.loc[df.iloc[val_index].index, 'Fold'] = fold_num
        
        new_train_dict = {'Mask':[], 'Gender':[], 'Age':[], 'Path':[], 'Label':[], 'Fold':[]}
        for i, path in tqdm.tqdm(enumerate(list(df['path']))):
            image_path = conf.train_dir + "/crop_images/" + path if crop else conf.train_dir + "/images/" + path
            temp = list(sorted(os.listdir(image_path)))
            for img in temp:
                if img[0] == '.': # ignore hidden files
                    continue
                elif 'normal' in img:
                    new_train_dict['Mask'] = new_train_dict['Mask'] + ['Not Wear']
                elif 'incorrect' in img:
                    new_train_dict['Mask'] = new_train_dict['Mask'] + ['Incorrect']
                else:
                    new_train_dict['Mask'] = new_train_dict['Mask'] + ['Wear']
                new_train_dict['Gender'] = new_train_dict['Gender'] + [df['gender'][i].capitalize()]
                new_train_dict['Age'] = new_train_dict['Age'] + [self.get_age(df['age'][i])]
                new_train_dict['Path'] = new_train_dict['Path'] + [image_path + "/" + img]
                label = self.get_label(new_train_dict['Mask'][-1], new_train_dict['Gender'][-1], new_train_dict['Age'][-1])
                new_train_dict['Label'] = new_train_dict['Label'] + [label]
                new_train_dict['Fold'] = new_train_dict['Fold'] + [df['Fold'][i]]
        
        self.train_df = pd.DataFrame(new_train_dict)
            
    def get_images(self):
        image_paths = [path for path in list(self.train_df['Path'])]
        images = []
        
        for path in image_paths:
            image = Image.open(path)
            images.append(image)

        return images
        
    def get_labels(self, opt):
        return list(self.train_df[opt])
    
    def convert_gender_age(self, gender, age):
        """
        gender와 age label을 조합하여 고유한 레이블을 만듭니다.
        이를 구하는 이유는 train/val의 성별 및 연령 분포를 맞추기 위함입니다. (by Stratified K-Fold)
        :param gender: `male` or `female`
        :param age: 나이를 나타내는 int.
        :return: gender & age label을 조합한 레이블
        """
        gender_label = 1 if 'female' else 0
        age_label = 0
        if 30 <= age < 58:
            age_label = 1
        if age >= 58:
            age_label = 2
        return gender_label * 3 + age_label
        
    def get_age(self, age) -> str:
        if age < 30:
            return '<30'
        elif 30 <= age < 58:
            return '>=30 and <60'
        else:
            return '>=60'
        
    def get_label(self, mask, gender, age) -> int:
        label = 0
        if mask == 'Incorrect':
            label += 6
        if mask == 'Not Wear':
            label += 12
        if gender == 'Female':
            label += 3
        if age == '>=30 and <60':
            label += 1
        if age == '>=60':
            label += 2
        return label   

    
class TransformedDataset(Dataset):
    def __init__(self, dataset, transform, fold_idx):
        self.train_dict = {'image': [], 'label': []}
        self.transform(dataset, transform, fold_idx)

    def __getitem__(self, index):
        return self.train_dict['image'][index]['image'], self.train_dict['label'][index]

    def __len__(self):
        return len(self.train_dict['image'])
    
    def transform(self, dataset, transform, fold_idx):
        for idx in list(map(int, dataset.train_df[dataset.train_df.Fold != fold_idx].index)):
            if transform:
                image = transform(image = np.asarray(dataset.__getitem__(idx)[0]))
            else:
                image = dataset.__getitem__(idx)[0]
            self.train_dict['image'].extend([image])
            self.train_dict['label'].extend([dataset.__getitem__(idx)[1]])
        

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image = np.asarray(image))
        return image

    def __len__(self):
        return len(self.img_paths)
