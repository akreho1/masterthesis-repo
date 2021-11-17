import os
import pathlib
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from ellipse import LsqEllipse
from math import pi
from matplotlib.patches import Ellipse
import Augmentor as aug

from tensorflow import keras
import numpy as np


class IMG_reader():
    def __init__(self, train_path, test_path, img_size, augmentacija, batch_size=1, *args, **kwargs):
        self.train_path=train_path
        self.test_path = test_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.augmentations=augmentacija

        train_image_names = []
        train_label_names = []
        test_image_names = []
        test_label_names = []

        if type(train_path) == list:
            for p in train_path:
                for file in os.listdir(p + '/slike/'):
                    if file.endswith(".JPG") or file.endswith(".JPEG") or file.endswith(".jpg"):
                        train_image_names.append(os.path.join(p + '/slike/', file))

        else:
            for file in os.listdir(train_path + '/slike/'):
                if file.endswith(".JPG") or file.endswith(".JPEG") or file.endswith(".jpg"):
                    train_image_names.append(os.path.join(train_path + '/slike/', file))
                    
        size=0            
        if type(test_path) == list:
            for p in test_path:
                if size>500:
                    break
                for file in os.listdir(p + '/slike/'):
                    if file.endswith(".JPG") or file.endswith(".JPEG") or file.endswith(".jpg"):
                        test_image_names.append(os.path.join(p + '/slike/', file))
                size+=1

        else:
            for file in os.listdir(test_path + '/slike/'):
                if size>500:
                    break
                if file.endswith(".JPG") or file.endswith(".JPEG") or file.endswith(".jpg"):
                    test_image_names.append(os.path.join(test_path + '/slike/', file))
                size+=1

        shit, self.val = train_test_split(train_image_names, train_size=0.8, test_size=0.2)
        self.train, self.test1 = train_test_split(shit, train_size=0.8, test_size=0.2)
        
        self.test2=test_image_names
        self.train_data = self.parse_image(self.train, augment=True)
        self.val_data = self.parse_image(self.val)
        self.test1_data, self.test1_labels = self.parse_image_test(self.test1)
        self.test2_data, self.test2_labels = self.parse_image_test(self.test2)


    def parse_image(self, img_list: list, augment=False):
        i = 0
        output = [[] for x in range(4)]
        images = []
        while True:
            for img_path in img_list:
                image = PIL.Image.open(img_path)
                image=image.convert('RGB')
                gray_image=image.convert('L')
                fn = lambda x: 1 if x>0 else 0
                
                label_path = img_path.replace('slike', 'maske_iris')
                if label_path.endswith('.JPG'):
                    label_path = label_path.replace('JPG', 'png')
                elif label_path.endswith('.JPEG'):
                    label_path = label_path.replace('JPEG', 'png')
                elif label_path.endswith('.jpg'):
                    label_path = label_path.replace('jpg', 'png')
                mask = PIL.Image.open(label_path)
                mask=mask.convert('L').point(fn, '1')
                
                if self.img_size is not None:
                    image = image.resize(self.img_size)
                    gray_image = gray_image.resize(self.img_size)
                    mask = mask.resize(self.img_size)
                    
                if augment and self.augmentations is not None:
                    image, gray_image, mask = self.augment(image, gray_image, mask)

                gray_image=np.asarray(gray_image)
                mask=np.asarray(mask)
                image=np.asarray(image)
                B_channel=image[:, :, 2].astype('float32')/255.0
                G_channel=image[:, :, 1].astype('float32')/255.0
                R_channel=image[:, :, 0].astype('float32')/255.0
                
                gray_image=gray_image[:, :, np.newaxis]
                mask=mask[:, :, np.newaxis]
                B_channel=B_channel[:, :, np.newaxis]
                G_channel=G_channel[:, :, np.newaxis]
                R_channel=R_channel[:, :, np.newaxis]

                output[0].append(mask)
                output[1].append(B_channel)
                output[2].append(G_channel)
                output[3].append(R_channel)
                images.append(gray_image)

                i += 1

                if i % self.batch_size == 0:
                    i = 0
                    a = [np.array(x) for x in output]
                    b = np.array(images)
                    output = [[] for x in range(4)]
                    images = []
                    yield b, a
                        
    def parse_image_test(self, img_list: list, augment=False):
        output = [[] for x in range(4)]
        images = []
        for img_path in img_list:
            image = PIL.Image.open(img_path)
            image=image.convert('RGB')
            gray_image=image.convert('L')
            fn = lambda x: 1 if x>0 else 0
            
            label_path = img_path.replace('slike', 'maske_iris')
            if label_path.endswith('.JPG'):
                label_path = label_path.replace('JPG', 'png')
            elif label_path.endswith('.JPEG'):
                label_path = label_path.replace('JPEG', 'png')
            elif label_path.endswith('.jpg'):
                label_path = label_path.replace('jpg', 'png')
            mask = PIL.Image.open(label_path)
            mask=mask.convert('L').point(fn, '1')
            
            if self.img_size is not None:
                image = image.resize(self.img_size)
                gray_image = gray_image.resize(self.img_size)
                mask = mask.resize(self.img_size)
                
            if augment and self.augmentations is not None:
                image, gray_image, mask = self.augment(image, gray_image, mask)

            gray_image=np.asarray(gray_image)
            mask=np.asarray(mask)
            image=np.asarray(image)
            B_channel=image[:, :, 2].astype('float32')/255.0
            G_channel=image[:, :, 1].astype('float32')/255.0
            R_channel=image[:, :, 0].astype('float32')/255.0
            
            gray_image=gray_image[:, :, np.newaxis]
            mask=mask[:, :, np.newaxis]
            B_channel=B_channel[:, :, np.newaxis]
            G_channel=G_channel[:, :, np.newaxis]
            R_channel=R_channel[:, :, np.newaxis]

            output[0].append(mask)
            output[1].append(B_channel)
            output[2].append(G_channel)
            output[3].append(R_channel)
            images.append(gray_image)
        return images, output   

    def augment(self, image: PIL.Image, mask_1: PIL.Image, mask_2: PIL.Image):
        augs = self.augmentations

        def flip(image, mask_1, mask_2):
            p = random.uniform(0, 1)
            if p < 0.3:
                image = image.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
                mask_1 = mask_1.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
                mask_2 = mask_2.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
            return image, mask_1, mask_2

        def rotate(image, mask_1, mask_2):
            
            p = random.uniform(0, 1)
            if p < 0.4:
                x = random.uniform(-1, 1)
                angle = x * 25
                image = image.rotate(angle)
                mask_1 = mask_1.rotate(angle)
                mask_2 = mask_2.rotate(angle)
            return image, mask_1, mask_2
            
        def translate(image, mask_1, mask_2):
            p=random.uniform(0, 1)
            if p<0.3:
                x=random.uniform(-1, 1)
                delta=x*20
                k=random.uniform(0, 1)
                if k>0.5:
                    image = image.transform(self.img_size, PIL.Image.AFFINE, (1, 0, delta, 0, 1, 0))
                    mask_1 = mask_1.transform(self.img_size, PIL.Image.AFFINE, (1, 0, delta, 0, 1, 0))
                    mask_2 = mask_2.transform(self.img_size, PIL.Image.AFFINE, (1, 0, delta, 0, 1, 0))
                else:
                    image = image.transform(self.img_size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, delta))
                    mask_1 = mask_1.transform(self.img_size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, delta))
                    mask_2 = mask_2.transform(self.img_size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, delta))
                    
            return image, mask_1, mask_2

        func = {'flip': flip, 'rotate': rotate, 'translate': translate}

        for aug in augs:
            image, mask_1, mask_2 = func[aug](image, mask_1, mask_2)

        return image, mask_1, mask_2
        
        
    def get_data(self):
        return (self.train_data, self.val_data, self.test1_data, self.test1_labels, self.test2_data, self.test2_labels)

    def get_train_size(self):
        return len(self.train)

    def get_val_size(self):
        return len(self.val)
        
    def get_test_size(self):
        return len(self.test1)
    
    def get_test2_size(self):
        return len(self.test2)