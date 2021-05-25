from albumentations import *
from albumentations.pytorch import ToTensorV2


def get_transforms(ver='simple', need=('train', 'val', 'test'), img_size=(224, 224), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    """
    train/validation/test set의 augmentation 함수를 정의
    
    Args:
        ver: 얼마나 다양한 augmentation 함수를 사용할 건지에 대한 옵션 - simple, hard로 구분
        need: 'train', 'val', 'test' 중 무엇에 대한 augmentation 함수를 얻을 건지에 대한 옵션
        img_size: Augmentation 이후 얻을 이미지 사이즈
        mean: 이미지를 Normalize할 때 사용될 RGB 평균값
        std: 이미지를 Normalize할 때 사용될 RGB 표준편차

    Returns:
        transformations: Augmentation 함수들이 저장된 dictionary
    """
    transformations = {}
    
    if ver == 'simple':
        if 'train' in need:
            print("I'm in simple train")
            transformations['train'] = Compose([
                CenterCrop(img_size[0], img_size[1]),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
        if 'val' in need:
            print("I'm in simple val")
            transformations['val'] = Compose([
                CenterCrop(img_size[0], img_size[1]),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
        if 'test' in need:
            transformations['test'] = Compose([
                CenterCrop(img_size[0], img_size[1]),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
            
    if ver == 'hard':
        if 'train' in need:
            print("I'm in train")
            transformations['train'] = Compose([
                CenterCrop(img_size[0], img_size[1]),
                CLAHE(p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
        if 'val' in need:
            print("I'm in val")
            transformations['val'] = Compose([
                CenterCrop(img_size[0], img_size[1]),
                CLAHE(p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
        if 'test' in need:
            transformations['test'] = Compose([
                CenterCrop(img_size[0], img_size[1]),
                CLAHE(p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
    
    return transformations