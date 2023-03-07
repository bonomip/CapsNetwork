import numpy as np
import scipy.io as spio
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import tensorflow as tf

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def _unpack(dataset):
    ans_set = dataset['affNISTdata']['label_int']
    img_set = dataset['affNISTdata']['image']
    trans = dataset['affNISTdata']['human_readable_transform']
    
    img_set = np.transpose(img_set)
    img_set = np.reshape(img_set, (-1, 40, 40))
    ans_set = ans_set.astype(np.uint8)

    return img_set, ans_set, trans

def _load_MNIST(train):
    s = "training" if train else "test"
    path = './data/affNIST/originals/'+s+'.mat'
    img_set, ans_set, trans = _unpack(loadmat(path))
    return img_set, ans_set

def convert_to_torch(img):
    rx = torch.from_numpy(img)
    rx = torch.unsqueeze(rx, 0)
    return rx

def convert_to_tensor(img):
    rx = tf.convert_to_tensor(img, dtype=tf.dtypes.float32)
    rx = np.reshape(rx, [1, 40, 40])
    return rx

def check_bounds(x, rx):
    a = np.sum(x)
    b = torch.sum(rx).item()
    c = abs(a - b)
    return c >= 1

def apply_random_translation(img, size=40, mode=transforms.InterpolationMode.BILINEAR):
    rx = convert_to_torch(img)
    
    flag = True;
    while flag:
        _rx = rx
        _rx = transforms.RandomAffine(0, translate=(0.6,0.6), interpolation=mode)(rx)
        flag = check_bounds(img, _rx)
        
    rx = convert_to_tensor(_rx)
    rx = tf.reshape(rx, [size, size])
    return rx

def create_random_mnist():

    print("Creating random mnist (train)... ")

    x, y = _load_MNIST(True)
    
    for i in range(0, x.shape[0]):
        x[i] = apply_random_translation(x[i])

    print("Saving dataset... ")
    np.save('./data/rMNIST/x_train_v1.npy', x)
    np.save('./data/rMNIST/y_train_v1.npy', y)

    print("Creating random mnist (test)... ")

    x, y = _load_MNIST(False)
    
    for i in range(0, x.shape[0]):
        x[i] = apply_random_translation(x[i])

    print("Saving dataset... ")
    np.save('./data/rMNIST/x_test_v1.npy', x)
    np.save('./data/rMNIST/y_test_v1.npy', y)

def load(train):
    s = 'train' if train else 'test'

    print("Load Custom rMNIST "+s+" dataset v1... ")

    X_ = np.load('./data/rMNIST/x_'+s+'_v1.npy')
    y_ = np.load('./data/rMNIST/y_'+s+'_v1.npy')
    
    return X_, y_





        