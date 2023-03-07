import numpy as np
import tensorflow as tf
from tqdm import tqdm

## project class import

from capsuleNetwork import CapsuleNetwork as CapsNet

import dataset.affNIST as affNIST
import dataset.caffNIST as caffNIST
import dataset.MNIST as MNIST
import dataset.cWSaffNIST as cWSaffNIST
import dataset.rMNIST as rMNIST
import dataset.sMNIST as sMNIST

class Setup:

    GEN = [
        "original_MNIST",
        "affNIST",
        "my_MNIST", # deprecated
        "Custom_affNIST", # deprecated
        "Custom_affNIST_without_shearing", # deprecated
        "rMNIST",
        "sMNIST"
    ] #dataset keys
    BATCH_SIZE = 64
    DEBUG_NO_MINI_BATCH = 10

    #architecture params
    params = {
        "size": 0,
        "no_of_conv_kernels": 256,
        "no_of_primary_caps_channels": 32,
        "no_of_secondary_capsules": 10,
        "primary_capsule_vector": 8,
        "secondary_capsule_vector": 16,
        "r":3,
    }

    def __init__(self, debug=False, no_gpu_check=False):
        self.debug = debug
        if not no_gpu_check:
            self.check_for_gpu()

############################## MODEL

    def init_model(self, id, version, x, y, learning_rate):
        self.params["id"] = id
        self.params["version"] = version
        self.params["size"] = 40 if (self.GEN.index(id) <= 5) else 28
        
        if(learning_rate > 0):
            self.params["learning_rate"] = learning_rate;
      
        model = CapsNet(**self.params)

        _ = model.train(x[:1],y[:1])
        return model

    def load_ckpt(self, model, epochs=-1):
        print("Loading model... ")
        model.load(epochs)
        return model

    def train_model(self, model, batch, epochs, resume=False, v_batch=0):
        print("Training model... ")
        model.train_for_epochs( batch, epochs, resume=resume, v_batch=v_batch)
        return model

############################## DATASET

    def switch_dataset(self, string, train, create, version):
        X_ = 0
        y_ = 0
        
        if string == self.GEN[0]: # toronto padded MNIST
            
            X_, y_ =affNIST.load_MNIST(train)
                    
        elif string == self.GEN[1]: # toronto affNIST
            
            (X_, y_) = affNIST.load(train)            

        if string == self.GEN[2]: # my padded MNIST
            
            X_, y_ = MNIST.load(train)

        elif string == self.GEN[3]: # CUSTOM AFFNIST
            
            if(create):
                
                (X_, y_) = caffNIST.create_custom_affnist(version, train)
                
            else:
                
                (X_, y_) = caffNIST.load(version, train)

        elif string == self.GEN[4]: # CUTOM AFFNIST NO SHEARING

            if(create):
                
                (X_, y_) = cWSaffNIST.create_custom_affnist_without_shearing(version, train)
                
            else:
                
                (X_, y_) = cWSaffNIST.load(version, train)

        elif string == self.GEN[5]: # RANDOM translation mnist
            if(create):
                
                (X_, y_) = rMNIST.create_random_mnist()
                
            else:
                
                (X_, y_) = rMNIST.load(train)
        
        elif string == self.GEN[6]: # shifted MNIST 28x28
            if(create):
                
                (X_, y_) = sMNIST.create_shifted_mnist()
                
            else:
                
                (X_, y_) = sMNIST.load(train)

        ### add others datasets

        return X_, y_

    def _process_data(self, x, y, shuffle):
        if ( self.debug ):
        
            x_ = x[:self.BATCH_SIZE*self.DEBUG_NO_MINI_BATCH]
            y_ = y[:self.BATCH_SIZE*self.DEBUG_NO_MINI_BATCH]
        
        else:
        
            x_ = x
            y_ = y

        x_ = x_ / 255.0
        x_ = tf.cast(x_, dtype=tf.float32)
        x_ = tf.expand_dims(x_, axis=-1)
        dataset = tf.data.Dataset.from_tensor_slices((x_, y_))
        if shuffle:

            dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
        
        dataset = dataset.batch(batch_size=self.BATCH_SIZE)
        return x_, y_, dataset       

    def load_data(self, id, train, version="_v1", create=False):
        (x, y) = self.switch_dataset(id, train, create, version)
        print("Processing data... ")
        x, y, batch = self._process_data(x, y, shuffle=train)

        return x, y, batch

############################# VALIDATION

    def get_accuracy(self, model, batch, no_img, description=""):
        training_sum = 0
        s = "Eval. accuracy "+description
        with tqdm(total=len(batch)) as pbar:
            pbar.set_description_str(s)
            for X_batch, y_batch in batch:
                        
                training_sum += sum(model.predict(X_batch)==y_batch.numpy())
                pbar.update(1)

        return training_sum/no_img

############################# GPU CHECK

    def check_for_gpu(self):
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            print(device_name)
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

############################ GET OBJECTS

    def get_model_patience():
        return CapsNet.patience

    def get_total_images(self, x):
        return x.shape[0]   
