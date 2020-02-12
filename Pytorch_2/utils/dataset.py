from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from skimage.transform import resize, rotate
from skimage.filters import gaussian
from skimage import io
import Augmentor
import cv2


class BasicDataset(Dataset):
    """
    imgs dir : 이미지 디렉토리 경로(path of image directory)
    masks_dir : 이미지에 대한 mask 디렉토리 경로(path of directory masks correspond to image )
    resize : 입력 이미지 크기 조절 사이즈(Unet paper based, 572)
    """
    def __init__(self, imgs_dir, masks_dir, resizing=572, is_transform=True, shuffle=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.resizing = resizing
        self.is_transform = is_transform
        self.shuffle=shuffle

        self.image_name_list = [splitext(file)[0] for file in listdir(imgs_dir) #이미지 파일 가져오고, 그중에서 이름만 가져오기 (img.jpg => extract img except for .jpg)
                    if not file.startswith('.')] # 폴더안에 존재하는 이미지들의 이름들의 리스트 (list of image names)
        logging.info(f'Creating dataset with {len(self.image_name_list)} examples') # 로그 기록
    
    """
    self.ids : 총 데이터셋의 크기를 나타낸다.(Size of total dataset)
    """
    def __len__(self):
        return len(self.image_name_list)

    """
    idx : range(0, len(self.ids)) 의 원소 중 하나. 0부터 차례대로 들어온다. 
    (one of element in range(0, len(self.ids))). input numbers in sequence.
    """
    
    def __getitem__(self, idx):
            
        image_name = self.image_name_list[idx]
        #glob(searching criteria) : find files that meet searching criteria, output type : list.
        #glob(탐색 기준식) : 탐색 기준식을 만족하는 파일을 찾아, 그 항목들을 리스트로 반환.
        mask_file = glob(self.masks_dir + image_name + '.*') # (in the case of 'image name == mask name')
        img_file = glob(self.imgs_dir + image_name + '.*') #(in the case of 'image name == mask name')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {image_name}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {image_name}: {img_file}'
        
        # print("img name : {}".format(img_file[0]))
        img = io.imread(img_file[0])
        mask_for_notf = io.imread(mask_file[0])
        mask = cv2.imread(mask_file[0]) # for augmenting, gray image: cv2.imread=(width, height, 3) , io.imread=(width, height)


        assert img.shape == mask.shape, \
            f'Image and mask {image_name} should be the same size, but are {img.size} and {mask.size}'

        if self.is_transform == False:
            img = self.preprocess(img, self.resizing)
            mask = self.preprocess(mask_for_notf, self.resizing)
            data = {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
        else:
            if np.random.uniform(size=1)[0] >= 0.7:
                  sigma = np.random.uniform(0.1,1,size=1)[0]
                  img = (gaussian(img, sigma=sigma, multichannel=True)*255).astype(np.uint8)

            p = Augmentor.DataPipeline([[img, mask]])
            p.resize(probability=1.0, width=256, height=256)
            p.rotate_without_crop(probability=0.3, max_left_rotation=25, max_right_rotation=25)
            p.shear(probability=0.3, max_shear_left=0.5, max_shear_right=0.5)
            p.flip_random(probability=0.6)
#           p.skew_tilt(probability=0.3, magnitude=0.1)
#           p.random_distortion(probability=0.3, grid_height=10, grid_width=10, magnitude=1)
#           p.zoom(probability=1.0, min_factor=1.1, max_factor=1.3)

            sample_p = p.sample(1)
            sample_p = np.array(sample_p).squeeze()

            p_img = sample_p[0]
            p_mask = sample_p[1]
            augmented_mask = (p_mask//255)*255 #np.where(p_mask<=0, p_mask, 255)

            q = Augmentor.DataPipeline([[p_img]])
            q.random_contrast(probability=0.3, min_factor=0.2, max_factor=1.0)  # low to High
            q.random_brightness(probability=0.3, min_factor=0.2, max_factor=1.0)  # dark to bright

            sample_q = q.sample(1)
            sample_q = np.array(sample_q).squeeze()

            augmented_img = sample_q
            augmented_mask = augmented_mask[::,::,0]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            augmented_mask = cv2.morphologyEx(augmented_mask, cv2.MORPH_CLOSE, kernel) 

            result_img= self.preprocess_wo_resize(augmented_img)
            result_mask = self.preprocess_wo_resize(augmented_mask)
            
            
            data = {'image': torch.from_numpy(result_img), 'mask': torch.from_numpy(result_mask)}
        return data

    @classmethod
    def preprocess(cls, img, resizing):
        img = resize(img, (resizing, resizing), anti_aliasing=True) # skimage.transform.resize

        img_nd = np.array(img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        # numpy image: H x W x C
        # torch image: C X H X W
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_wo_resize(cls, img):
        img_nd = np.array(img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=0) #(H, W) -> (H, W)
            return img_nd

        # HWC to CHW
        # numpy image: H x W x C
        # torch image: C X H X W
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans