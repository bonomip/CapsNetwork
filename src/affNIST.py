import scipy.io as spio
import numpy as np

#test set is 10k per batch, trainning set is 50k per batch

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

def load(train):
    s = "training" if train else "test"

    print("Load AffNIST "+s+" dataset... ")

    path = './data/affNIST/'+s+'_batches/1.mat'
    dataset = loadmat(path) 

    ans_set = dataset['affNISTdata']['label_int']
    test_set = dataset['affNISTdata']['image']

    test_set = np.transpose(test_set)
    test_set = np.reshape(test_set, (-1, 40, 40))
    ans_set = ans_set.astype(np.uint8)

    return test_set,ans_set