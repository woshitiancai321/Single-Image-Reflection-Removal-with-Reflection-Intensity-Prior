import os.path
from os.path import join
from data.image_folder import make_dataset
from data.transforms import Sobel, to_norm_tensor, to_tensor, ReflectionSythesis_1, ReflectionSythesis_2
from PIL import Image
import random
import torch
import math

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import util.util as util
import data.torchdata as torchdata
import cv2
import numpy as np
def reflect_strenght(raw_img,reflect_img):
    raw_img=raw_img.permute(1,2,0)
    raw_img = raw_img.numpy()
    reflect_img=reflect_img.permute(1,2,0)
    reflect_img = reflect_img.numpy()
    gray_lap_all=[]

    pppp=raw_img*255
    rrrr=reflect_img*255
    pppp=cv2.cvtColor(pppp,cv2.COLOR_BGR2GRAY)
    rrrr=cv2.cvtColor(rrrr,cv2.COLOR_BGR2GRAY)
    height, width = pppp.shape[:2]
    new_height = height // 7
    new_width = width // 7

    # 存储子图像均值的列表
    means = []

    # 分割原始图像为4*4的子图像
    for i in range(7):
        for j in range(7):
            # 计算当前子图像在原始图像中的位置
            x_start = i * new_height
            x_end = x_start + new_height
            y_start = j * new_width
            y_end = y_start + new_width

            # 获取当前子图像
            sub_ground_image = pppp[x_start:x_end, y_start:y_end]
            sub_reflect_image = rrrr[x_start:x_end, y_start:y_end]
            # 计算当前子图像的均值
            sub_ground_mean = np.mean(sub_ground_image)
            sub_reflect_mean = np.mean(sub_reflect_image)
            sub_image_mean=sub_reflect_mean/(sub_reflect_mean+sub_ground_mean+1e-6)
            sub_image_mean = np.clip(sub_image_mean, 0, 1,dtype=np.float32)
            # if(sub_image_mean<0):
            #     sub_image_mean=0
            # 将当前子图像的均值存储到列表中
            sub_image_mean = torch.tensor(sub_image_mean).unsqueeze(0)
            means.append(sub_image_mean)
            #means.append(sub_image_mean)
    means = torch.cat(means, dim=0)
    return means
def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    
    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    target_size = int(random.randint(224, 448) / 2.) * 2
    # target_size = int(random.randint(256, 480) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (224,224))
    # i, j, h, w = get_params(img_1, (256,256))
    img_1 = F.crop(img_1, i, j, h, w)
    
    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)
    
    return img_1,img_2


BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()


class CEILDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        super(CEILDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms

        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = self.paths[:size]

        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma, low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2
        self.B_paths = self.paths[0:num_paths]
        self.R_paths = self.paths[num_paths:2*num_paths]

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)
        syn_model = self.syn_model
        t_img, r_img, m_img = syn_model(t_img, r_img)
        
        B = to_tensor(t_img)
        R = to_tensor(r_img)
        M = to_tensor(m_img)

        return B, R, M
        
    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)
        
        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]
        
        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')
    
        B, R, M = self.data_synthesis(t_img, r_img)
        #reflect_t=reflect_strenght(B,R)
        B=B*2-1
        R=R*2-1
        M=M*2-1
        fn = os.path.basename(B_path)
        return {'SR': M, 'HR': B, 'fn': fn, 'real':True,'R': R}
    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))

class CEILrealTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None):
        super(CEILrealTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        #fn1=fn.replace("_M_","_T_")
        #fn1=fn.replace("m","g")
        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)
        t_img = t_img.resize((224, 224))
        m_img = m_img.resize((224, 224))
        B = to_tensor(t_img)
        M = to_tensor(m_img)
        #reflect ratio
        R=M-B
        #reflect_t=reflect_strenght(B,R)
        M=M*2-1
        B=B*2-1
        R=R*2-1
        #dic =  {'SR': M, 'HR': B, 'fn': fn, 'real':True, 'R': reflect_t} # fake reflection gt 
        # if self.flag is not None:
        #     dic.update(self.flag)
        return {'SR': M, 'HR': B, 'fn': fn, 'real':True, 'R': R}

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
class CEILTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None):
        super(CEILTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        if "_M" in fn:
            fn1=fn.replace("_M","_T")
            fn2=fn.replace("_M","_R")
        elif "mixture" in fn:
            fn1=fn.replace("mixture","background")
            fn2=fn.replace("mixture","reflection")
        else:
            fn1=fn.replace("m","g")
            fn2=fn.replace("m","r")        
        t_img = Image.open(join(self.datadir, 'transmission_layer', fn1)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'reflection', fn2)).convert('RGB')
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)
        t_img = t_img.resize((224, 224))
        m_img = m_img.resize((224, 224))
        r_img = r_img.resize((224, 224))
        B = to_tensor(t_img)
        M = to_tensor(m_img)
        R = to_tensor(r_img)
        #R=M-B
        #reflect_t=reflect_strenght(B,R)
        R=R*2-1
        M=M*2-1
        B=B*2-1
        return {'SR': M, 'HR': B, 'fn': fn, 'real':True, 'R': R}

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
class CEILTestDatasetETRI(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None):
        super(CEILTestDatasetETRI, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        if "_M" in fn:
            fn1=fn.replace("_M","_T")
            fn2=fn.replace("_M","_R")
        elif "mixture" in fn:
            fn1=fn.replace("mixture","background")
            fn2=fn.replace("mixture","reflection")
        else:
            fn1=fn.replace("m","g")
            fn2=fn.replace("m","r")        
        t_img = Image.open(join(self.datadir, 'blended', fn1)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'blended', fn2)).convert('RGB')
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)
        t_img = t_img.resize((224, 224))
        m_img = m_img.resize((224, 224))
        r_img = r_img.resize((224, 224))
        B = to_tensor(t_img)
        M = to_tensor(m_img)
        R = to_tensor(r_img)
        #R=M-B
        #reflect_t=reflect_strenght(B,R)
        R=R*2-1
        M=M*2-1
        B=B*2-1
        return {'SR': M, 'HR': B, 'fn': fn, 'real':True, 'R': R}

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1./len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' %(self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio/residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index%len(dataset)]
            residual -= ratio
    
    def __len__(self):
        return self.size
