import os #for accessing the file system of the system
from skimage import io
import cv2
import numpy as np
import keras
import Augmentor

"""
def iter_sequence_infinite(seq):
    '''Iterate indefinitely over a Sequence.
    # Arguments
        seq: Sequence object
    # Returns
        Generator yielding batches.
    '''
    while True:
        for item in seq:
            yield item
"""
            

# data generator class
class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, imgs_dir, masks_dir, batch_size=10, img_size=128, n_classes=1, n_channels=3, shuffle=True):
        self.id_names = ids
        self.indexes = np.arange(len(self.id_names))
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    # for printing the statistics of the function
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.id_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, id_name):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        img_path = os.path.join(self.imgs_dir, id_name)  # polyp segmentation/images/id_name.jpg
        mask_path = os.path.join(self.masks_dir, id_name) # polyp segmenatation/masks/id_name.jpg

        img = io.imread(img_path)
        mask = cv2.imread(mask_path)

        p = Augmentor.DataPipeline([[img, mask]])
        p.resize(probability=1.0, width=self.img_size, height=self.img_size)
        p.rotate_without_crop(probability=0.3, max_left_rotation=10, max_right_rotation=10)
        #p.random_distortion(probability=0.3, grid_height=10, grid_width=10, magnitude=1)
        p.shear(probability=0.3, max_shear_left=1, max_shear_right=1)
        #p.skew_tilt(probability=0.3, magnitude=0.1)
        p.flip_random(probability=0.3)

        sample_p = p.sample(1)
        sample_p = np.array(sample_p).squeeze()

        p_img = sample_p[0]
        p_mask = sample_p[1]
        augmented_mask = (p_mask // 255) * 255  # denoising

        q = Augmentor.DataPipeline([[p_img]])
        q.random_contrast(probability=0.3, min_factor=0.2, max_factor=1.0)  # low to High
        q.random_brightness(probability=0.3, min_factor=0.2, max_factor=1.0)  # dark to bright

        sample_q = q.sample(1)
        sample_q = np.array(sample_q).squeeze()

        image = sample_q
        mask = augmented_mask[::, ::, 0]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        """
        # reading the image from dataset
        ## Reading Image
        image = io.imread(img_path)  # reading image to image vaiable
        image = resize(image, (self.img_size, self.img_size), anti_aliasing=True)  # resizing input image to 128 * 128

        mask = io.imread(mask_path, as_gray=True)  # mask image of same size with all zeros
        mask = resize(mask, (self.img_size, self.img_size), anti_aliasing=True)  # resizing mask to fit the 128 * 128 image
        mask = np.expand_dims(mask, axis=-1)
        """

        # image normalization
        image = image / 255.0
        mask = mask / 255.0

        return image, mask

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.id_names) / self.batch_size))

    def __getitem__(self, index):  # index : batch no.
        # Generate indexes of the batch
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.id_names[k] for k in indexes]

        imgs = list()
        masks = list()

        for id_name in batch_ids:
            img, mask = self.__data_generation__(id_name)
            imgs.append(img)
            masks.append(np.expand_dims(mask,-1))

        imgs = np.array(imgs)
        masks = np.array(masks)

        return imgs, masks  # return batch
