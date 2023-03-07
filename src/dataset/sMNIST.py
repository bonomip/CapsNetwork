import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import numpy as np
import tensorflow as tf

def _load_MNIST(train):
    a = b = c = d = 0
    s = "training" if train else "test"
    
    print("Load MINST "+s+" dataset from keras... ")
    (a, b), (c , d)= tf.keras.datasets.mnist.load_data()

    if(train):
        
        X_ = a
        y_ = b
        
    else:
        
        X_ = c
        y_ = d

    return X_, y_

def convert_to_torch(img):
    rx = torch.from_numpy(img)
    rx = torch.unsqueeze(rx, 0)
    return rx

def convert_to_tensor(img):
    rx = tf.convert_to_tensor(img, dtype=tf.dtypes.float32)
    rx = np.reshape(rx, [1, 28, 28])
    return rx

def check_bounds(x, rx):
    a = np.sum(x)
    b = torch.sum(rx).item()
    c = abs(a - b)
    return c >= 1

def apply_random_pixel_shift(img, mode=transforms.InterpolationMode.BILINEAR):
    rx = convert_to_torch(img)
    
    flag = True;
    while flag:
        _rx = rx
        # pixel_shift = img_width * a 
        # 2 = 28 * a
        # 2/28 = a
        # a = 0.07143
        _rx = transforms.RandomAffine(0, translate=(0.07143,0.07143), interpolation=mode)(rx)
        flag = check_bounds(img, _rx)
        
    rx = convert_to_tensor(_rx)
    rx = tf.reshape(rx, [28, 28])
    return rx    
    
def create_shifted_mnist():
    print("Creating shifted mnist (train)... ")

    x, y = _load_MNIST(True)
    
    for i in range(0, x.shape[0]):
        x[i] = apply_random_pixel_shift(x[i])

    print("Saving dataset... ")
    np.save('./data/sMNIST/x_train_v1.npy', x)
    np.save('./data/sMNIST/y_train_v1.npy', y)

    print("Creating shifted mnist (test)... ")

    x, y = _load_MNIST(False)
    
    for i in range(0, x.shape[0]):
        x[i] = apply_random_pixel_shift(x[i])

    print("Saving dataset... ")
    np.save('./data/sMNIST/x_test_v1.npy', x)
    np.save('./data/sMNIST/y_test_v1.npy', y)
        
def load(train):
    s = 'train' if train else 'test'

    print("Load Custom shifted MNIST "+s+" dataset v1... ")

    X_ = np.load('./data/sMNIST/x_'+s+'_v1.npy')
    y_ = np.load('./data/sMNIST/y_'+s+'_v1.npy')
    
    return X_, y_