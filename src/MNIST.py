import numpy as np
import tensorflow as tf

def from28to40(images):
    return np.pad(images, ((0, 0),(6, 6),(6, 6)) )

def load(train):
    a = b = c = d = 0
    s = "training" if train else "test"
    
    print("Load MINST "+s+" dataset from keras... ")
    (a, b), (c , d)= tf.keras.datasets.mnist.load_data()

    if(train):
        
        X_ = from28to40(a)
        y_ = b
        
    else:
        
        X_ = from28to40(c)
        y_ = d

    return X_, y_