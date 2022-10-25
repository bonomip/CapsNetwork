import numpy as np
import tensorflow as tf
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import torch
import math
from tqdm import tqdm

def get_pads(i, j, size):
    pad_i = (size-i)/2
    if (pad_i%2) != 0 :
        pad_i_s = int(math.floor(pad_i))
        pad_i_f = int(math.ceil(pad_i))
    else:
        pad_i_s = int(pad_i)
        pad_i_f = int(pad_i)
        
    pad_j = (size-j)/2
    if (pad_j%2) != 0 :
        pad_j_s = int(math.floor(pad_j))
        pad_j_f = int(math.ceil(pad_j))
    else:
        pad_j_s = int(pad_j)
        pad_j_f = int(pad_j)
    
    return pad_i_s, pad_i_f, pad_j_s, pad_j_f

def check_bounds(img, size):
    for i in range(0, size):
        if ( img[0][i][0] > 0 ) or ( img[0][i][size-1] > 0 ) or ( img[0][0][i] > 0 ) or ( img[0][size-1][i] > 0 ) :
            return False
        
def apply_random_affine_transformation(img, size):
    rotation = 20.0
    shear = (0, 0, 0, 0)
    scaling = [0.8, 1.2]
    translation = 0 # no ink can fall outside
        
    rx = np.reshape(img, [1, 28, 28])
    rx = torch.from_numpy(rx)
            
    #APPLY SCALING
    i = math.ceil( np.random.uniform(28*scaling[0], 28*scaling[1]) )
    j = math.ceil( np.random.uniform(28*scaling[0], 28*scaling[1]) )
    rx = fn.resize(rx, size=[ i, j ], interpolation=transforms.InterpolationMode.NEAREST)
    
    ## APPLY PADDING
    pad_i_s, pad_i_f, pad_j_s, pad_j_f = get_pads(i, j, size)
    rx = np.pad(rx[0], ( ( pad_i_s, pad_i_f ), ( pad_j_s, pad_j_f ) ) )
    
    ## RESHAPE
    rx = np.reshape(rx, [1, 40, 40])
    
    ## APPLY OTHER TRANSFORMATION
    rx = torch.from_numpy(rx)
    
    flag = True
    while flag: 
        tx = transforms.RandomAffine(rotation, (0, 0.1), (1, 1), shear)
        rx_ = tx(rx)
        flag = check_bounds(rx_, size)
    
    rx_ =  tf.convert_to_tensor(rx_, dtype=tf.dtypes.float32)

    return rx_ 

def apply_affine(tensor, description):
    #shape (60000, 28, 28)
    
    n = tensor.shape[0]
    images = list()
    
    with tqdm(total=(n/100)) as pbar:
        
        pbar.set_description_str(description)
        
        for i in range(0, n):
            img = apply_random_affine_transformation(tensor[i, :, :], 40)
            images.append(img[0])
            if i % 100 == 0:
                pbar.update(1)
        
        pbar.set_postfix_str("")
        
    result = tf.stack(images)
    return result

def create_custom_affnist_without_shearing(train, version, r=3):
    
    print("Load MIST dataset from keras... ")
    (a, b), (c , d)= tf.keras.datasets.mnist.load_data()
    
    if train:
     
        print("Apply affine transformations no shearing... ")
        X_ = apply_affine(a, "Apply transformation on train set")
        y_ = b
        
        for _ in range(r-1):
            
            X_temp = apply_affine(a, "Apply transformation on train set")
            X_ = np.concatenate( (X_, X_temp), axis=0 )
            y_ = np.concatenate( (y_, b), axis=0)
        
        print("Saving dataset... ")
        np.save('./data/caffNIST_without_shearing/x_train'+version+'.npy', X_)
        np.save('./data/caffNIST_without_shearing/y_train'+version+'.npy', y_)
    else:
        
        y_ = d
        print("Apply affine transformations no shearing... ")
        X_ = apply_affine(c, "Apply transformation on test set")
        print("Saving dataset... ")
        np.save('./data/caffNIST_without_shearing/x_test'+version+'.npy', X_)
        np.save('./data/caffNIST_without_shearing/y_test'+version+'.npy', y_)

    return X_, y_

def load(version, train):
    s = 'train' if train else 'test'

    print("Load Custom affNIST without shearing "+s+" dataset "+version+"... ")

    X_ = np.load('./data/caffNIST_without_shearing/x_'+s+version+'.npy')
    y_ = np.load('./data/caffNIST_without_shearing/y_'+s+version+'.npy')
    
    return X_, y_